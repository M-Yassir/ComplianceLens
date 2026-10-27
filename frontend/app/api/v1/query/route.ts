import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Reads runtime environment variable injected by Docker Compose
    const backendUrl = process.env.INTERNAL_API_URL || "http://api:8000";

    const backendRes = await fetch(`${backendUrl}/api/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!backendRes.ok) {
      const errorText = await backendRes.text();
      console.error(`[Proxy Error] Backend returned status ${backendRes.status}:`, errorText);
      return NextResponse.json(
        { error: `Backend API returned status ${backendRes.status}`, details: errorText },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("[Proxy Exception] Failed to reach FastAPI container:", error.message);
    return NextResponse.json(
      { error: "Failed to connect to backend service", details: error.message },
      { status: 500 }
    );
  }
}