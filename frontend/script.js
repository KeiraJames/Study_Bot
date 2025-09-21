async function uploadFile() {
    const file = document.getElementById("upload").files[0];
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/upload", {method:"POST", body: formData});
    const data = await res.json();
    alert("Document loaded!");
}

async function askQuestion() {
    const question = document.getElementById("question").value;
    const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
    });
    const data = await res.json();
    document.getElementById("answer").innerText = data.answer;
}
