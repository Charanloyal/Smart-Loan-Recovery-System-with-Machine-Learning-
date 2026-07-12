let f1Chart = null;

function switchTab(tabName) {
    // Reset alerts
    hideAlert();
    
    // Toggle active tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Toggle active panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Activate target
    if (tabName === 'predict') {
        event.currentTarget.classList.add('active');
        document.getElementById('panel-predict').classList.add('active');
    } else {
        event.currentTarget.classList.add('active');
        document.getElementById('panel-metrics').classList.add('active');
        loadMetrics(); // Fetch and render charts/tables
    }
}

function showAlert(message, type = 'error') {
    const errorAlert = document.getElementById('alert-error');
    const errorText = document.getElementById('alert-error-text');
    
    errorText.innerText = message;
    errorAlert.className = `alert ${type} active`;
}

function hideAlert() {
    const errorAlert = document.getElementById('alert-error');
    if (errorAlert) {
        errorAlert.classList.remove('active');
    }
}

async function handleEvaluation(event) {
    event.preventDefault();
    hideAlert();
    
    const submitBtn = document.getElementById('submit-btn');
    const resultBox = document.getElementById('result-box');
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Running Assessment...';
    
    // Get form values
    const payload = {
        credit_score: parseInt(document.getElementById('credit_score').value),
        annual_income: parseFloat(document.getElementById('annual_income').value),
        loan_amount: parseFloat(document.getElementById('loan_amount').value),
        interest_rate: parseFloat(document.getElementById('interest_rate').value),
        debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
        employment_length: parseInt(document.getElementById('employment_length').value),
        home_ownership: document.getElementById('home_ownership').value,
        loan_purpose: document.getElementById('loan_purpose').value
    };
    
    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            let errorMsg = data.detail || 'Prediction failed';
            if (Array.isArray(errorMsg)) {
                errorMsg = errorMsg.map(err => err.msg).join(', ');
            }
            throw new Error(errorMsg);
        }
        
        // Update prediction details
        const defaultProbPercent = (data.default_probability * 100).toFixed(1) + '%';
        const recoveryRatePercent = (data.predicted_recovery_rate * 100).toFixed(1) + '%';
        
        document.getElementById('default-prob-value').innerText = defaultProbPercent;
        
        const riskTierValue = document.getElementById('risk-tier-value');
        riskTierValue.innerText = data.risk_tier;
        
        // Remove old risk classes and add appropriate color coding
        riskTierValue.className = '';
        if (data.risk_tier === 'Low') riskTierValue.classList.add('risk-low');
        else if (data.risk_tier === 'Medium') riskTierValue.classList.add('risk-medium');
        else if (data.risk_tier === 'High') riskTierValue.classList.add('risk-high');
        else if (data.risk_tier === 'Critical') riskTierValue.classList.add('risk-critical');
        
        document.getElementById('recovery-rate-value').innerText = recoveryRatePercent;
        document.getElementById('recovery-amount-value').innerText = '$' + data.expected_recovery_amount.toLocaleString(undefined, { minimumFractionDigits: 2 });
        document.getElementById('recommended-action').innerText = data.recommended_action;
        
        // Apply color styles to recommended action
        const recAction = document.getElementById('recommended-action');
        recAction.className = '';
        if (data.recommended_action === 'Auto-Approve') recAction.style.color = '#4ade80';
        else if (data.recommended_action === 'Manual Review') recAction.style.color = '#fbbf24';
        else if (data.recommended_action === 'Flagged Default / Collateral Hold') recAction.style.color = '#f97316';
        else recAction.style.color = '#ef4444';
        
        resultBox.classList.add('active');
        
    } catch (err) {
        resultBox.classList.remove('active');
        showAlert(err.message, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            Evaluate Risk & Recovery
        `;
    }
}

async function loadMetrics() {
    try {
        const response = await fetch('/api/v1/metrics');
        if (!response.ok) {
            throw new Error('Failed to fetch model performance metrics.');
        }
        
        const data = await response.json();
        
        // Render classification model comparison summary
        const bestClfName = data.best_classification;
        const bestClfVal = data.classification[bestClfName].f1_score;
        document.getElementById('best-clf').innerText = bestClfName;
        document.getElementById('best-clf-f1').innerText = `F1-Score: ${(bestClfVal * 100).toFixed(1)}%`;
        
        const bestRegName = data.best_regression;
        const bestRegVal = data.regression[bestRegName].r2_score;
        document.getElementById('best-reg').innerText = bestRegName;
        document.getElementById('best-reg-r2').innerText = `R² Score: ${bestRegVal.toFixed(3)}`;
        
        // Populate regression metrics table
        const tbody = document.getElementById('metrics-tbody');
        tbody.innerHTML = '';
        
        Object.keys(data.regression).forEach(name => {
            const metrics = data.regression[name];
            tbody.innerHTML += `
                <tr>
                    <td><strong>${name}</strong></td>
                    <td>${metrics.r2_score.toFixed(4)}</td>
                    <td>$${metrics.mae.toFixed(4)}</td>
                    <td>$${metrics.rmse.toFixed(4)}</td>
                </tr>
            `;
        });
        
        // Render Chart.js Chart
        renderF1Chart(data.classification);
        
    } catch (err) {
        showAlert(err.message, 'error');
    }
}

function renderF1Chart(classMetrics) {
    const labels = Object.keys(classMetrics);
    const f1Scores = labels.map(name => classMetrics[name].f1_score * 100);
    
    if (f1Chart) {
        f1Chart.destroy();
    }
    
    const ctx = document.getElementById('f1-chart').getContext('2d');
    f1Chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'F1-Score (%)',
                data: f1Scores,
                backgroundColor: [
                    '#aa7c11', // Dark Gold
                    '#d4af37', // Metallic Gold
                    '#ebd080'  // Bright Soft Gold
                ],
                borderWidth: 0,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#9e9a8f',
                        font: {
                            family: 'Outfit'
                        }
                    },
                    grid: {
                        color: 'rgba(212, 175, 55, 0.05)'
                    }
                },
                x: {
                    ticks: {
                        color: '#f7f6f0',
                        font: {
                            family: 'Outfit',
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}
