document.getElementById("verifyButton").addEventListener("click", () => {
    // Collect signature data (from canvas or file upload)
    let formData = new FormData();
    
    const canvasData = canvas.toDataURL(); // For canvas-based signature

    if (canvasData) {
        formData.append("signatureData", canvasData); // Append canvas data as base64 string
    }

    // If file uploaded, append file to the formData
    const fileInput = document.getElementById("fileInput");
    if (fileInput.files[0]) {
        formData.append("signature", fileInput.files[0]); // Append uploaded file
    }

    // Send data to the backend for verification
    fetch("http://localhost:3000/verify-signature", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Show verification result
        document.getElementById("verificationResult").style.display = "block";
        document.getElementById("resultMessage").textContent = 
            data.status === "success" ? "Signature verified successfully." : "Signature verification failed.";
        document.getElementById("confidenceScore").textContent = `${data.confidence}%`;
    })
    .catch(error => {
        console.error("Error verifying signature:", error);
        alert("Error verifying signature.");
    });
});
