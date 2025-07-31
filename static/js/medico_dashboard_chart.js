// static/js/medico_dashboard_chart.js

document.addEventListener('DOMContentLoaded', function() {
    const chartCanvas = document.getElementById('diagnosticosChart');
    
    // Se o elemento canvas existir na página...
    if (chartCanvas) {
        // ...e se ele tiver os dados que esperamos...
        if (chartCanvas.dataset.labels && chartCanvas.dataset.values) {
            try {
                // JSON.parse transforma a string de volta em um array JavaScript
                const labels = JSON.parse(chartCanvas.dataset.labels);
                const values = JSON.parse(chartCanvas.dataset.values);
                
                if (labels.length > 0) {
                    const ctx = chartCanvas.getContext('2d');
                    new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: '# de Laudos',
                                data: values,
                                backgroundColor: [
                                    'rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)',
                                    'rgba(255, 206, 86, 0.8)', 'rgba(75, 192, 192, 0.8)',
                                    'rgba(153, 102, 255, 0.8)', 'rgba(255, 159, 64, 0.8)'
                                ],
                                borderColor: '#fff',
                                borderWidth: 2
                            }]
                        },
                        options: { 
                            responsive: true, 
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                }
                            }
                        }
                    });
                }
            } catch (e) {
                console.error("Erro ao processar dados do gráfico:", e);
                chartCanvas.outerHTML = "<p class='text-danger text-center'>Erro ao carregar o gráfico.</p>";
            }
        }
    }
});