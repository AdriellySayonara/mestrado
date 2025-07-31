// static/js/results_dashboard_charts.js

document.addEventListener('DOMContentLoaded', function() {
    // Função genérica para criar os gráficos de boxplot comparativos
    function createComparativeBoxPlot(canvasId, fullChartData, metricKey, metricLabel) {
        const canvas = document.getElementById(canvasId);
        const metricData = fullChartData[metricKey];

        if (!canvas) {
            console.error(`Canvas com ID '${canvasId}' não encontrado.`);
            return;
        }

        if (!metricData || metricData.length === 0 || metricData.every(d => !d || d.length === 0)) {
            canvas.parentElement.innerHTML = `<div class="text-center p-5"><p class="text-muted">Dados insuficientes para o gráfico de ${metricLabel}.</p></div>`;
            return;
        }

        const shortLabels = fullChartData.labels.map((_, index) => `Comb. ${String.fromCharCode(65 + index)}`);
        
        new Chart(canvas.getContext('2d'), {
            type: 'boxplot',
            data: {
                labels: shortLabels,
                datasets: [{
                    label: metricLabel,
                    data: metricData,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    itemStyle: 'circle',
                    itemRadius: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                return fullChartData.labels[index].split('\n');
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: metricLabel },
                    },
                    x: {
                       title: { display: true, text: 'Combinação de Pipeline (ver legenda)'}
                    }
                }
            }
        });
    }

    // --- Lógica Principal ---
    const dataElement = document.getElementById('chart-data');
    if (dataElement) {
        try {
            const fullChartData = JSON.parse(dataElement.textContent);
            
            // Só tenta renderizar se houver pelo menos 2 combinações para comparar
            if (fullChartData && fullChartData.labels && fullChartData.labels.length > 1) {
                createComparativeBoxPlot('chartAcuracia', fullChartData, 'acuracia', 'Acurácia');
                createComparativeBoxPlot('chartSensibilidade', fullChartData, 'sensibilidade', 'Sensibilidade');
                createComparativeBoxPlot('chartEspecificidade', fullChartData, 'especificidade', 'Especificidade');
                createComparativeBoxPlot('chartAucRoc', fullChartData, 'auc_roc', 'AUC-ROC');
                createComparativeBoxPlot('chartKappa', fullChartData, 'kappa', 'Kappa');
            }
        } catch (e) {
            console.error("Erro ao renderizar os gráficos comparativos:", e);
        }
    }
});