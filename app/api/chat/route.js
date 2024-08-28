import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `
You are a helpful agent designed to assist students in finding professors based on their specific queries. For each student query, your task is to search and retrieve relevant professor data using a retrieval-augmented generation (RAG) approach. You will then provide the top three professors who best match the student's query. Each recommendation should include:

The professor's name
The department or subject they teach
The university or college they are associated with
A brief summary of their ratings, including helpfulness, teaching quality, and difficulty
Additional insights like whether they are known for being approachable, providing extra resources, or being research-oriented (based on available data)
Your goal is to ensure that each response is concise, accurate, and helpful to students seeking professor recommendations.

When making recommendations, focus on:

Matching the professor’s expertise with the subject or query
Including only up-to-date and relevant information
Providing diverse options to help students make an informed decision
`;

export async function POST(req) {
  try {
    const data = await req.json();
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index("rag").namespace("ms1");
    const genAI = new GoogleGenerativeAI(process.env.GEMENI_API_KEY);

    const text = data[data.length - 1].content;

    const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
    const embeddingResult = await embeddingModel.embedContent(text);

    let embedding;
    if (Array.isArray(embeddingResult.embedding)) {
      embedding = embeddingResult.embedding;
    } else if (
      embeddingResult.embedding &&
      Array.isArray(embeddingResult.embedding.values)
    ) {
      embedding = embeddingResult.embedding.values;
    } else {
      throw new Error("Unexpected embedding format");
    }

    // Ensure the embedding is an array of numbers
    if (embedding.some((value) => typeof value !== "number")) {
      throw new Error("Invalid embedding format: not all values are numbers");
    }

    const results = await index.query({
      topK: 5,
      includeMetadata: true,
      vector: embedding,
    });

    let resultString = "";
    results.matches.forEach((match) => {
      resultString += `
            Returned Results:
            Professor: ${match.id}
            Review: ${match.metadata.stars}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}
            \n\n`;
    });

    // Use Gemini for chat completion
    const chatModel = genAI.getGenerativeModel({ model: "gemini-pro" });
    const chat = chatModel.startChat({
      history: [
        { role: "user", parts: [{ text: systemPrompt }] },
        ...data.slice(0, -1).map((msg) => ({
          role: msg.role === "assistant" ? "model" : "user",
          parts: [{ text: msg.content }],
        })),
      ],
    });

    const result = await chat.sendMessageStream(data[data.length - 1].content);

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of result.stream) {
            const chunkText = chunk.text();
            controller.enqueue(encoder.encode(chunkText));
          }
        } catch (error) {
          controller.error(error);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);
  } catch (error) {
    console.error("Error in POST function:", error);
    return new NextResponse(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
