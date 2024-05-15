
import OpenAI from "openai";
import apiKey from './apiKey.js';
import fs from 'fs'

const openai = new OpenAI({
    apiKey
});

async function main() {
    const response = await openai.images.generate({
        model: "dall-e-3",
        prompt: "刘亦菲微笑着和我打招呼，在一个阳光明媚的早晨。",
        n: 1,
        size: "1024x1024",
    });
    console.log('response', response)
}

main();