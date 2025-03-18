document.getElementById("trainModelBtn").addEventListener("click", function () {
    fetch("/train", { method: "POST" })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error("Error:", error));
});

document.getElementById("verifyBtn").addEventListener("click", function () {
    let genuineFile = document.getElementById("genuineUpload").files[0];
    let forgedFile = document.getElementById("forgedUpload").files[0];

    if (!genuineFile || !forgedFile) {
        alert("Please upload both images.");
        return;
    }

    let formData = new FormData();
    formData.append("genuine", genuineFile);
    formData.append("forged", forgedFile);

    fetch("/verify", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Result: ${data.result} (Score: ${data.score.toFixed(2)})`;
    })
    .catch(error => console.error("Error:", error));
});

// Function to Preview Images
function previewImage(event, imgId) {
    let reader = new FileReader();
    reader.onload = function () {
        document.getElementById(imgId).src = reader.result;
    };
    reader.readAsDataURL(event.target.files[0]);
}
