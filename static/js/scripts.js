function validateForm() {
    const fileInput = document.getElementById('file');
    const modelName = document.getElementById('model_name').value.trim();
    const description = document.getElementById('description').value.trim();
    
    if (!modelName) {
        alert('Model name is required.');
        return false;
    }
    if (!description) {
        alert('Description is required.');
        return false;
    }
    if (!fileInput.files.length) {
        alert('Please upload a CSV file.');
        return false;
    }
    if (!fileInput.files[0].name.endsWith('.csv')) {
        alert('Only CSV files are allowed.');
        return false;
    }
    return true;
}