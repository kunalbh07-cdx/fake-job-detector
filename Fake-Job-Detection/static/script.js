const textarea = document.getElementById("job_text");
const charCount = document.getElementById("charCount");

if (textarea) {
    textarea.addEventListener("input", () => {
        charCount.textContent = textarea.value.length + " characters";
    });
}

// Reset everything 
function resetPage() {
    textarea.value = "";
    charCount.textContent = "0 characters";

    const resultBox = document.getElementById("resultBox");
    if (resultBox) resultBox.remove();

    const bar = document.getElementById("confidenceFill");
    if (bar) bar.remove();
}

// Set confidence bar width safely (NO inline CSS)
window.onload = () => {
    const bar = document.getElementById("confidenceFill");
    if (bar) {
        const confidence = bar.getAttribute("data-confidence");
        bar.style.width = confidence + "%";
    }
};
