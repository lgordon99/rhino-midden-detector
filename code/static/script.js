let batch = 0;
let batch_identifiers = [];
let current_index = 0;
let current_modality = 'thermal';
let current_zoom = 'in';
let batch_labels = [];
let intervalID = setInterval(() => checkBatchExists(batch), 5000); // poll the server every 5 seconds
const spans = document.getElementsByTagName('span');


function adjustButtonSize() {
    const imgDim = document.getElementsByTagName('img')[0].offsetWidth;
    const settingButtons = document.getElementsByClassName('setting-button');

    for (let i = 0; i < settingButtons.length; i++) {
        settingButtons[i].style.height = `${0.5*imgDim}px`;
    }

    for (let i = 0; i < spans.length; i++) {
        spans[i].style.width = `${0.5*imgDim}px`;
    }

    const labelButtons = document.getElementsByClassName('label-button');

    for (let i = 0; i < labelButtons.length; i++) {
        labelButtons[i].style.width = `${0.5*imgDim}px`;
    }
}

function adjustFontSize() {
    const buttons = document.getElementsByTagName('button');
    
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].offsetHeight < buttons[i].offsetWidth) {
            const buttonHeight = buttons[i].offsetHeight;
            buttons[i].style.fontSize = `${0.6*buttonHeight}px`;    
        } else {
            const buttonWidth = buttons[i].offsetWidth;
            buttons[i].style.fontSize = `${0.6*buttonWidth}px`;    
        }
    }
    
    for (let i = 0; i < spans.length; i++) {
        const spanWidth = spans[i].offsetHeight;
        spans[i].style.fontSize = `${0.6*spanWidth}px`;
    }
}

function initiate() {
    window.addEventListener('resize', function() {
        adjustButtonSize();
        adjustFontSize();
    });    
}

function setButtonVisibility(value) {
    let buttons = document.getElementsByTagName('button');
    
    for (let i = 0; i < buttons.length; i++) {
        buttons[i].style.visibility = value;
    }
}

function checkBatchExists() {
    fetch(`/check_batch_exists/${batch}`)
        .then(response => response.json())
        .then(data => {
            if (data.batch_path_exists) {
                batch_identifiers = data.batch_identifiers;
                document.getElementById('batch message').innerText = `Batch ${batch}: Image ${current_index+1}/${batch_identifiers.length}, ID = ${batch_identifiers[current_index]}`;
                document.getElementsByTagName('img')[0].style.visibility = 'visible';
                showImage();
                adjustButtonSize();
                adjustFontSize();
                setButtonVisibility('visible');
                clearInterval(intervalID); // stop checking for the path
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function getOtherModality(){
    if (current_modality == 'thermal') {
        return 'rgb';
    } 
    return 'thermal';
}

function getOtherZoom(){
    if (current_zoom == 'in') {
        return 'out';
    } 
    return 'in';
}

function setModality(modality) {
    current_modality = modality;
}

function setZoom(zoom) {
    current_zoom = zoom;
}

function showImage() {
    document.getElementsByTagName('img')[0].src = `static/images/batch-${batch}-images/img-${batch_identifiers[current_index]}-${current_modality}-${current_zoom}.png`;
    document.getElementById(current_modality).style.backgroundColor = 'green';
    document.getElementById(getOtherModality()).style.backgroundColor = 'gray';
    document.getElementById(current_zoom).style.backgroundColor = 'green';
    document.getElementById(getOtherZoom()).style.backgroundColor = 'gray';
}

function next() {
    current_index += 1;
    
    if (current_index < batch_identifiers.length) { // next image
        current_modality = 'thermal';
        current_zoom = 'in';
        showImage();
        document.getElementById('batch message').innerText = `Batch ${batch}: Image ${current_index+1}/${batch_identifiers.length}, ID = ${batch_identifiers[current_index]}`;    
    } else { // next batch
        batch_identifiers_string = '';
        batch_labels_string = '';

        for (let i = 0; i < batch_identifiers.length; i++) {
            batch_identifiers_string += `${batch_identifiers[i]}`;
            batch_labels_string += `${batch_labels[i]}`;

            if (i < batch_identifiers.length - 1) {
                batch_identifiers_string += ',';
                batch_labels_string += ',';
            }
        }
        
        console.log(batch_identifiers_string);
        console.log(batch_labels_string);
        fetch(`/save_batch_labels/${batch}/${batch_identifiers_string}/${batch_labels_string}`);
        batch += 1;
        batch_identifiers = [];
        current_index = 0;
        batch_labels = [];
        document.getElementById('batch message').innerText = `Waiting for batch ${batch}...`
        setButtonVisibility('hidden');
        document.getElementsByTagName('img')[0].style.visibility = 'hidden';
        intervalID = setInterval(() => checkBatchExists(batch), 5000); // poll the server every 5 seconds
    }
}

function middenLabel() {
    batch_labels.push(1);
    next();
}

function emptyLabel() {
    batch_labels.push(0);
    next();
}