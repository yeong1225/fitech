navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    const video = document.getElementById('webcam');
    video.srcObject = stream;
    video.onloadedmetadata = function(e) {
      video.play();
      updateCanvas();
    };
  })
  .catch(err => {
    console.log("An error occurred: " + err);
  });

function updateCanvas() {
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');

  // 사각형의 크기를 정의
  
  const rectWidth = 400;
  const rectHeight = 720;

  // 사각형의 중앙 위치를 계산
  const rectX = (canvas.width - rectWidth) / 2;
  const rectY = (canvas.height - rectHeight) / 2;

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
  }

  function draw() {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the video
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw the more opaque black rectangles around the white rectangle
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';

    // Top rectangle
    ctx.fillRect(0, 0, canvas.width, rectY);
    // Bottom rectangle
    ctx.fillRect(0, rectY + rectHeight, canvas.width, canvas.height - rectY - rectHeight);
    // Left rectangle
    ctx.fillRect(0, rectY, rectX, rectHeight);
    // Right rectangle
    ctx.fillRect(rectX + rectWidth, rectY, canvas.width - rectX - rectWidth, rectHeight);

    // Draw the rounded white rectangle border
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 4;
    roundRect(ctx, rectX, rectY, rectWidth, rectHeight, 10); // Adjust the radius as needed
    ctx.stroke();

    requestAnimationFrame(draw);
  }

  draw();
}



