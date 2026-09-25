const copyButton = document.getElementById("copy-bibtex");
const citationEntry = document.getElementById("bibtex-entry");
const citationStatus = document.getElementById("citation-status");

if (copyButton && citationEntry && citationStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citationEntry.textContent.trim());
      citationStatus.textContent = "BibTeX copied to clipboard.";
    } catch {
      citationStatus.textContent = "Copy is unavailable. Select the citation or download the .bib file.";
    }
  });
}
