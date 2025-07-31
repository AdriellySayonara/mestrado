// static/js/train_form_logic.js (Versão Definitiva)

document.addEventListener('DOMContentLoaded', function() {
    console.log("O script train_form_logic.js foi carregado.");

    // --- SELETORES DE ELEMENTOS ---
    // É crucial que os IDs e nomes aqui correspondam exatamente ao seu HTML e formulário.
    const cnnSection = document.getElementById('cnn_options_section');
    const classicClassifierSection = document.getElementById('classic_classifier_section');
    const smoteOptionsDiv = document.getElementById('smote_options');

    const pipelineRadios = document.querySelectorAll('input[name="pipeline_type"]');
    const augmentationClassicRadios = document.querySelectorAll('input[name="augmentation_method_classic"]');
    
    // --- FUNÇÃO PRINCIPAL DE VISIBILIDADE ---
    function updateFormVisibility() {
        console.log("Atualizando a visibilidade do formulário...");

        // Garante que os elementos de rádio existem antes de tentar ler
        const selectedPipelineRadio = document.querySelector('input[name="pipeline_type"]:checked');
        if (!selectedPipelineRadio) {
            console.error("Nenhum 'pipeline_type' selecionado.");
            return;
        }
        const selectedPipeline = selectedPipelineRadio.value;
        
        // Lógica para seções principais
        if (cnnSection) {
            cnnSection.style.display = (selectedPipeline !== 'classic_only') ? 'block' : 'none';
        }
        if (classicClassifierSection) {
            classicClassifierSection.style.display = (selectedPipeline !== 'end_to_end') ? 'block' : 'none';
        }

        // --- LÓGICA ESPECÍFICA DO SMOTE ---
        // Esta é a parte mais importante para o seu problema atual
        if (smoteOptionsDiv) {
            const selectedAugmentationRadio = document.querySelector('input[name="augmentation_method_classic"]:checked');
            if (selectedAugmentationRadio) {
                const augmentationMethod = selectedAugmentationRadio.value;
                console.log("Método de balanceamento selecionado:", augmentationMethod);
                
                if (augmentationMethod === 'smote') {
                    smoteOptionsDiv.style.display = 'block';
                    console.log("Mostrando opções do SMOTE.");
                } else {
                    smoteOptionsDiv.style.display = 'none';
                    console.log("Escondendo opções do SMOTE.");
                }
            }
        } else {
            console.warn("Elemento com ID 'smote_options' não foi encontrado no HTML.");
        }
    }

    // --- OUVINTES DE EVENTOS (TRIGGERS) ---
    // Uma lista de todos os controles que devem disparar a função de atualização
    const controlsToWatch = [
        ...pipelineRadios,
        ...augmentationClassicRadios
    ];

    controlsToWatch.forEach(control => {
        if (control) {
            control.addEventListener('change', updateFormVisibility);
        }
    });

    // Roda a função uma vez na carga inicial da página para garantir o estado correto
    updateFormVisibility();
});