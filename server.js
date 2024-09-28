const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.post('/api/detect', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        const formData = new FormData();
        formData.append('image', fs.createReadStream(imagePath));

        const response = await axios.post('http://localhost:8080/detect', formData, {
            headers: formData.getHeaders(),
        });

        fs.unlinkSync(imagePath);

        res.status(200).json(response.data);
    } catch (error) {
        console.error('Error during disease detection:', error);
        res.status(500).json({ error: 'Error during disease detection' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Express server berjalan di port ${PORT}`);
});
