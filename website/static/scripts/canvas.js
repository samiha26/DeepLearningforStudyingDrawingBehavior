const canvas = document.getElementById("canvas");
const canvasContainer = document.getElementsByClassName("canvas-container")[0];
canvas.height = canvasContainer.clientHeight;
canvas.width = canvasContainer.clientWidth;

const ctx = canvas.getContext("2d");
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let prevX = null;
let prevY = null;

ctx.lineWidth = 2;
ctx.strokeStyle = "black";

let draw = false;
let started = false;

// Erasing the canvas with white color
let eraseBtn = document.getElementById("erase-btn");
eraseBtn.addEventListener("click", function() {
    eraseBtn.innerHTML = eraseBtn.innerHTML == "Erase" ? "Draw" : "Erase";
    if (ctx.strokeStyle == "#000000") {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 20;
    }
    else{
        ctx.strokeStyle = "black";
        ctx.lineWidth = 2;
    }
});
// Saving drawing as image
let nextBtn = document.getElementById("next-btn");
nextBtn.addEventListener("click", () => {
    let dataURL = canvas.toDataURL("image/png");
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/save_drawing", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                // Handle successful response from the server
                let response = JSON.parse(xhr.responseText);
                console.log(response.message);
                // Redirect to the next page based on the response
                if (response.nextPage) {
                    window.location.href = response.nextPage;
                
                } else {
                    console.error("Next page not specified in the server response");
                }
            } else {
                console.error("Error occurred while saving the drawing:", xhr.status);
            }
        }
    };
    xhr.send(JSON.stringify({ "image": dataURL }));
});

canvas.addEventListener("mousedown", (e) => {draw = (true && started)});
canvas.addEventListener("mouseup", (e) => draw = false);

canvas.addEventListener("mousemove", (e) => {
    if(prevX == null || prevY == null || !draw){
        prevX = e.clientX;
        prevY = e.clientY;
        return;
    }
    let currentX = e.clientX;
    let currentY = e.clientY;
    
    let rect = canvas.getBoundingClientRect();
    let left_add = rect.left;
    let top_add = rect.top;

    ctx.beginPath();
    ctx.moveTo(prevX-left_add, prevY-top_add);
    ctx.lineTo(currentX-left_add, currentY-top_add);
    ctx.stroke();

    prevX = currentX;
    prevY = currentY;
});

let startBtn = document.getElementById("start-btn");
let time_left = document.getElementsByClassName("time-left")[0];
let time_slider = document.getElementsByClassName("time-slider")[0];
let duration = 60;

startBtn.addEventListener("click", () => {
    timer = setInterval(() => {
        if (duration <= 0) {
            started = false;
            duration = 60;
            time_left.innerHTML = duration;
            startBtn.classList.remove("disabled");
            time_slider.style.width = "100%";
            clearInterval(timer);
            return;
        }
        started = true;
        duration--;
        time_left.innerHTML = duration;
        time_slider.style.width = `${duration/60*100}%`;
    }, 1000);
    startBtn.classList.add("disabled");
});

let clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
});

let resetBtn = document.getElementById("reset-btn");
resetBtn.addEventListener("click", () => {
    started = false;
    duration = 60;
    time_left.innerHTML = duration;
    startBtn.classList.remove("disabled");
    time_slider.style.width = "100%";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    clearInterval(timer);
});
