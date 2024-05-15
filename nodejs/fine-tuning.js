import OpenAI from "openai";
import apiKey from './apiKey.js';
import fs from 'fs'

const openai = new OpenAI({
    apiKey
});

async function main() {
    const response = await openai.chat.completions.create({
        model: "ft:gpt-3.5-turbo-1106:personal::9NBPhPCV",
        // model: "gpt-3.5-turbo",
        messages: [
          {
            "role": "system",
            "content": "你是一个问答机器人，回答尽可能让对方喜欢，要体现你的高情商"
          },
          {
            "role": "user",
            "content": "你觉得我胖吗？"
          }
        ],
        temperature: 0.8,
        max_tokens: 128,
        top_p: 1,
    });
    console.log(response.choices[0])
}

main();