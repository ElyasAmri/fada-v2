import { ChatApp } from "@/components/chat-app";

export default function Home() {
  const endpoint =
    process.env.NEXT_PUBLIC_LLM_ENDPOINT ??
    process.env.LLM_ENDPOINT ??
    "https://elyasamri-fada-gemma4-zerogpu.hf.space";
  const model =
    process.env.NEXT_PUBLIC_LLM_MODEL ??
    process.env.LLM_MODEL ??
    "google/gemma-4-E2B-it + elyasamri/gemma-4-e2b-fada-adapter";
  const apiKey =
    process.env.NEXT_PUBLIC_LLM_API_KEY ??
    process.env.LLM_API_KEY ??
    process.env.HF_TOKEN ??
    null;

  return <ChatApp endpoint={endpoint} model={model} apiKey={apiKey} />;
}
