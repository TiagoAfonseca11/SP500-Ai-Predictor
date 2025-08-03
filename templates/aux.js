// Variables for interval selection
let isSelecting = false;
let selectionStart = null;
let selectionEnd = null;
let selectionOverlay = null;

function updateChart(data) {
    const canvas = document.getElementById('priceChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart
    if (priceChart) {
        priceChart.destroy();
    }
    
    // Prepare data
    const prices = data.prices || [];
    const dates = data.dates || [];
    const predictions = data.predictions || [];
    
    if (prices.length === 0) {
        // Show no data message
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('No chart data available', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.05)');
    
    // Chart configuration
    const config = {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'S&P 500 Price',
                data: prices,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: 'rgb(59, 130, 246)',
                pointHoverBorderColor: 'white',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#f8fafc',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            return `Price: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 8
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            },
            elements: {
                point: {
                    hoverRadius: 6
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            onHover: function(event, elements) {
                canvas.style.cursor = 'crosshair';
            }
        }
    };
    
    // Create new chart
    priceChart = new Chart(ctx, config);
    
    // Add interval selection functionality
    setupIntervalSelection(canvas, prices, dates);
}

function setupIntervalSelection(canvas, prices, dates) {
    // Remove existing event listeners
    canvas.removeEventListener('mousedown', handleMouseDown);
    canvas.removeEventListener('mousemove', handleMouseMove);
    canvas.removeEventListener('mouseup', handleMouseUp);
    canvas.removeEventListener('mouseleave', handleMouseLeave);
    
    function handleMouseDown(event) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        
        isSelecting = true;
        selectionStart = getDataIndexFromX(x);
        selectionEnd = selectionStart;
        
        // Remove existing overlay
        removeSelectionOverlay();
    }
    
    function handleMouseMove(event) {
        if (!isSelecting) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        
        selectionEnd = getDataIndexFromX(x);
        
        // Update selection overlay
        updateSelectionOverlay(prices, dates);
    }
    
    function handleMouseUp(event) {
        if (!isSelecting) return;
        
        isSelecting = false;
        
        // Show selection info for a moment
        setTimeout(() => {
            removeSelectionOverlay();
        }, 3000);
    }
    
    function handleMouseLeave(event) {
        if (isSelecting) {
            isSelecting = false;
            removeSelectionOverlay();
        }
    }
    
    function getDataIndexFromX(x) {
        const chartArea = priceChart.chartArea;
        const dataWidth = chartArea.right - chartArea.left;
        const dataCount = prices.length;
        
        if (x < chartArea.left) return 0;
        if (x > chartArea.right) return dataCount - 1;
        
        const relativeX = x - chartArea.left;
        const index = Math.round((relativeX / dataWidth) * (dataCount - 1));
        
        return Math.max(0, Math.min(dataCount - 1, index));
    }
    
    // Add event listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
}

function updateSelectionOverlay(prices, dates) {
    if (selectionStart === null || selectionEnd === null) return;
    
    const startIndex = Math.min(selectionStart, selectionEnd);
    const endIndex = Math.max(selectionStart, selectionEnd);
    
    if (startIndex === endIndex) return;
    
    const startPrice = prices[startIndex];
    const endPrice = prices[endIndex];
    const priceChange = endPrice - startPrice;
    const percentChange = (priceChange / startPrice) * 100;
    
    // Determine color based on change
    const isPositive = priceChange >= 0;
    const overlayColor = isPositive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
    const borderColor = isPositive ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)';
    
    // Get chart area coordinates
    const chartArea = priceChart.chartArea;
    const dataWidth = chartArea.right - chartArea.left;
    const dataCount = prices.length;
    
    const startX = chartArea.left + (startIndex / (dataCount - 1)) * dataWidth;
    const endX = chartArea.left + (endIndex / (dataCount - 1)) * dataWidth;
    
    // Remove existing overlay
    removeSelectionOverlay();
    
    // Create overlay element
    const overlay = document.createElement('div');
    overlay.className = 'selection-overlay';
    overlay.style.cssText = `
        position: absolute;
        left: ${Math.min(startX, endX)}px;
        top: ${chartArea.top}px;
        width: ${Math.abs(endX - startX)}px;
        height: ${chartArea.bottom - chartArea.top}px;
        background: ${overlayColor};
        border-left: 2px solid ${borderColor};
        border-right: 2px solid ${borderColor};
        pointer-events: none;
        z-index: 10;
        transition: all 0.1s ease;
    `;
    
    // Create info tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'selection-tooltip';
    tooltip.style.cssText = `
        position: absolute;
        left: ${Math.max(startX, endX) + 10}px;
        top: ${chartArea.top + 10}px;
        background: rgba(30, 41, 59, 0.95);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        pointer-events: none;
        z-index: 11;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    
    tooltip.innerHTML = `
        <div style="color: ${isPositive ? '#10b981' : '#ef4444'}; margin-bottom: 4px;">
            ${isPositive ? '↗' : '↘'} ${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%
        </div>
        <div style="color: #94a3b8; font-size: 11px;">
            $${startPrice.toFixed(2)} → $${endPrice.toFixed(2)}
        </div>
        <div style="color: #94a3b8; font-size: 11px; margin-top: 2px;">
            ${dates[startIndex]} to ${dates[endIndex]}
        </div>
    `;
    
    // Add to chart container
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.style.position = 'relative';
    chartContainer.appendChild(overlay);
    chartContainer.appendChild(tooltip);
    
    selectionOverlay = { overlay, tooltip };
}

function removeSelectionOverlay() {
    if (selectionOverlay) {
        if (selectionOverlay.overlay && selectionOverlay.overlay.parentNode) {
            selectionOverlay.overlay.parentNode.removeChild(selectionOverlay.overlay);
        }
        if (selectionOverlay.tooltip && selectionOverlay.tooltip.parentNode) {
            selectionOverlay.tooltip.parentNode.removeChild(selectionOverlay.tooltip);
        }
        selectionOverlay = null;
    }
    
    // Also remove any orphaned overlays
    const chartContainer = document.querySelector('.chart-container');
    const overlays = chartContainer.querySelectorAll('.selection-overlay, .selection-tooltip');
    overlays.forEach(el => el.remove());
}