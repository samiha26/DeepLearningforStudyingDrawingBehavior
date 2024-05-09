let nextBtn = document.getElementById("next-btn");
nextBtn.addEventListener("click", () => {
    fileName = (document.getElementsByTagName("title")[0].innerText.split(" ")[0]).toLowerCase();
    if (fileName == "house")
        window.location.href = "/result-tree.html";
    else if (fileName == "tree")
        window.location.href = "/result-person.html";
    else
        window.location.href = "/index.html";
});