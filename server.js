const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const app = express();
const upload = multer({ dest: 'uploads/' });


// Endpoint untuk scan gambar
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


// Endpoint untuk menambahkan artikel
app.post('/api/articles', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;
    const { title, content, author, publishDate, tags } = req.body;

    try {
        const formData = new FormData();
        formData.append('image', fs.createReadStream(imagePath));
        formData.append('title', title);
        formData.append('content', content);
        formData.append('author', author);
        formData.append('publishDate', publishDate);
        formData.append('tags', tags);

        // Kirim permintaan ke Flask server (port 8080)
        const response = await axios.post('http://localhost:8080/articles', formData, {
            headers: formData.getHeaders(),
        });

        fs.unlinkSync(imagePath);

        res.status(200).json(response.data);
    } catch (error) {
        console.error('Error posting article:', error);
        res.status(500).json({ error: 'Error posting article' });
    }
});


// Endpoint untuk mendapatkan semua artikel
app.get('/api/articles', async (req, res) => {
    try {
        // Kirim permintaan GET ke Flask server (port 8080)
        const response = await axios.get('http://localhost:8080/articles');
        res.status(200).json(response.data);
    } catch (error) {
        console.error('Error retrieving articles:', error);
        res.status(500).json({ error: 'Error retrieving articles' });
    }
});

// Endpoint untuk mendapatkan artikel berdasarkan ID
app.get('/api/articles/:id', async (req, res) => {
    const articleId = req.params.id;
    try {
        // Kirim permintaan GET ke Flask server (port 8080) berdasarkan ID
        const response = await axios.get(`http://localhost:8080/articles/${articleId}`);
        res.status(200).json(response.data);
    } catch (error) {
        if (error.response && error.response.status === 404) {
            res.status(404).json({ error: 'Artikel tidak ditemukan' });
        } else {
            console.error('Error retrieving article:', error);
            res.status(500).json({ error: 'Error retrieving article' });
        }
    }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Express server berjalan di port ${PORT}`);
});