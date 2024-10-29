const sharp = require('sharp');

function preprocessImage(imagePath) {
    return sharp(imagePath)
        .resize(640, 640) 
        .toBuffer();
}

module.exports = { preprocessImage };
