import copy

import torch

from action_dit import ActionExpertIDM


def _build_tiny_expert():
    return ActionExpertIDM(
        dim=32,
        video_dim=48,
        num_heads=4,
        num_layers=2,
        ffn_dim=64,
        freq_dim=16,
        action_dim=14,
        max_action_len=5,
        text_context_dim=16,
        joint_state_dim=14,
        pred_target="x0",
        use_rope=True,
        proprio_mode="text",
    ).train()


def _run(model, checkpointing):
    torch.manual_seed(7)
    noisy_actions = torch.randn(2, 5, 14, requires_grad=True)
    video_layer_feats = [
        torch.randn(2, 6, 48, requires_grad=True),
        torch.randn(2, 6, 48, requires_grad=True),
    ]
    output = model(
        noisy_actions=noisy_actions,
        timestep=torch.tensor([0.2, 0.7]),
        video_layer_feats=video_layer_feats,
        text_context=torch.randn(2, 3, 16),
        joint_state=torch.randn(2, 14),
        cond_timestep=torch.tensor([0.1, 0.4]),
        return_x0=True,
        use_gradient_checkpointing=checkpointing,
    )
    output.square().mean().backward()
    return (
        output.detach(),
        noisy_actions.grad.detach(),
        [feat.grad.detach() for feat in video_layer_feats],
        [param.grad.detach() for param in model.parameters() if param.grad is not None],
    )


def test_action_expert_checkpoint_matches_plain_forward_and_backward():
    plain = _build_tiny_expert()
    checkpointed = copy.deepcopy(plain)

    plain_result = _run(plain, checkpointing=False)
    checkpointed_result = _run(checkpointed, checkpointing=True)

    torch.testing.assert_close(plain_result[0], checkpointed_result[0])
    torch.testing.assert_close(plain_result[1], checkpointed_result[1])
    for plain_grad, checkpointed_grad in zip(
        plain_result[2] + plain_result[3],
        checkpointed_result[2] + checkpointed_result[3],
    ):
        torch.testing.assert_close(plain_grad, checkpointed_grad)
