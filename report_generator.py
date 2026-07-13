import os
from openai import OpenAI

from ui import NULL_REPORTER


def generate_report(transcript: str, video_path: str, client: OpenAI, reporter=NULL_REPORTER) -> str:
    """Generate a comprehensive report from the German transcript using GPT-4o.
    Returns the path to the generated report file."""

    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(video_dir, f"{video_name}_report.md")

    reporter.step("report", "Analyzing transcript & writing report (GPT-4o)", total=None)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert analyst. You will receive a transcript of a video in German.
Analyze the content thoroughly and produce a comprehensive report IN ENGLISH with the following structure:

# Video Analysis Report

## Topic
(What is the main topic/title of this discussion?)

## Executive Summary
(2-3 sentence overview of the content)

## Key Points Discussed
(Detailed bulleted list of all major points discussed, with sub-bullets for supporting details)

## Action Items & Conclusions
(Any action items, recommendations, or conclusions drawn - if applicable)

## Notable Quotes & Statements
(Significant quotes or statements, translated to English, with context)

Be thorough and capture all important information from the transcript."""
            },
            {
                "role": "user",
                "content": f"Here is the German transcript to analyze:\n\n{transcript}"
            }
        ],
        temperature=0.4,
        max_tokens=4000
    )

    report = response.choices[0].message.content

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    reporter.done("report")
    reporter.success(f"Analysis report → {report_path}")
    return report_path
