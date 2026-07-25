const API_URL = 'http://localhost:5000/api';  // Change for production

let currentMethod = 'linear';
let currentCoeffs = null;
let currentGraph = null;

function setMethod(method) {
    currentMethod = method;
    document.querySelectorAll('.method-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('btn-' + method).classList.add('active');
    document.getElementById('degreeBox').style.display = method === 'polynomial' ? 'block' : 'none';
}

async function solve() {
    const data = document.getElementById('dataInput').value;
    const degree = document.getElementById('polyDegree').value;
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        const response = await fetch(`${API_URL}/solve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method: currentMethod,
                data: data,
                degree: parseInt(degree)
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert(result.error);
            return;
        }
        
        currentCoeffs = result.coefficients;
        currentGraph = result.graph;
        
        document.getElementById('equation').innerHTML = result.equation;
        document.getElementById('steps').innerHTML = result.steps.join('<br>');
        
        drawGraph(result.graph, data);
        
        document.getElementById('results').style.display = 'block';
        document.getElementById('predictResult').style.display = 'none';
        
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

async function predictY() {
    const xVal = parseFloat(document.getElementById('predictX').value);
    if (isNaN(xVal)) {
        alert('Enter a valid number!');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method: currentMethod,
                coefficients: currentCoeffs,
                x: xVal
            })
        });
        
        const result = await response.json();
        document.getElementById('predictValue').innerHTML = `y = ${result.y.toFixed(6)}`;
        document.getElementById('predictResult').style.display = 'block';
        
    } catch (err) {
        alert('Prediction error: ' + err.message);
    }
}

function drawGraph(graphData, rawData) {
    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const padding = 50;
    
    // Parse original points
    const points = rawData.trim().split('\n').map(line => {
        const parts = line.split(',').map(s => parseFloat(s.trim()));
        return {x: parts[0], y: parts[1]};
    });
    
    let minX = Math.min(...points.map(p => p.x));
    let maxX = Math.max(...points.map(p => p.x));
    let minY = Math.min(...points.map(p => p.y), ...graphData.y);
    let maxY = Math.max(...points.map(p => p.y), ...graphData.y);
    
    const rangeX = maxX - minX;
    minX -= rangeX * 0.1; maxX += rangeX * 0.1;
    const rangeY = maxY - minY;
    minY -= rangeY * 0.2; maxY += rangeY * 0.2;
    
    const sx = x => padding + (x - minX) / (maxX - minX) * (w - 2 * padding);
    const sy = y => h - padding - (y - minY) / (maxY - minY) * (h - 2 * padding);
    
    // Clear
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, w, h);
    
    // Grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const x = padding + i * (w - 2 * padding) / 5;
        const y = padding + i * (h - 2 * padding) / 5;
        ctx.beginPath(); ctx.moveTo(x, padding); ctx.lineTo(x, h - padding); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(w - padding, y); ctx.stroke();
    }
    
    // Axes
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(padding, h - padding); ctx.lineTo(w - padding, h - padding); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padding, padding); ctx.lineTo(padding, h - padding); ctx.stroke();
    
    // Curve
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < graphData.x.length; i++) {
        const px = sx(graphData.x[i]);
        const py = sy(graphData.y[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    
    // Points
    ctx.fillStyle = '#ef4444';
    for (let p of points) {
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'white'; ctx.lineWidth = 2;
        ctx.stroke();
    }
}

setMethod('linear');