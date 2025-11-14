import os
import asyncio
from PIL import Image
from pyrogram import Client, filters, enums
from utils.misc import modules_help, prefix
from utils.scripts import format_exc
from utils.config import gemini_key
import google.generativeai as genai

genai.configure(api_key=gemini_key)
MODEL_NAME = "gemini-2.0-flash"
COOK_GEN_CONFIG = {
    "temperature": 0.35,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

def _valid_file(reply, file_type=None):
    if file_type == "image":
        return getattr(reply, "photo", None) is not None
    if file_type in {"audio", "video"}:
        return any(getattr(reply, attr, False) for attr in ("audio", "voice", "video", "video_note"))
    return (
        getattr(reply, "photo", None)
        or getattr(reply, "audio", None)
        or getattr(reply, "voice", None)
        or getattr(reply, "video", None)
        or getattr(reply, "video_note", None)
        or getattr(reply, "document", None)
    )

async def _upload_file(file_path, file_type):
    uploaded_file = genai.upload_file(file_path)
    while uploaded_file.state.name == "PROCESSING":
        await asyncio.sleep(5)
        uploaded_file = genai.get_file(uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise ValueError(f"{file_type.capitalize()} failed to process")
    return uploaded_file

async def prepare_input_data(reply, file_path, prompt):
    if reply.photo:
        with Image.open(file_path) as img:
            img.verify()
        return [prompt, Image.open(file_path)]
    if reply.video or reply.video_note:
        return [prompt, await _upload_file(file_path, "video")]
    if reply.audio or reply.voice:
        return [await _upload_file(file_path, "audio"), prompt]
    if reply.document and file_path.endswith(".pdf"):
        return [prompt, await _upload_file(file_path, "PDF")]
    if reply.document:
        return [await _upload_file(file_path, "document"), prompt]
    raise ValueError("Unsupported file type")

async def ai_process_handler(message, prompt, show_prompt=False, cook_mode=False, expect_type=None, status_msg="Processing..."):
    reply = message.reply_to_message

    if not reply:
        usage = f"Usage: {prefix}{message.command[0]} [prompt] [Reply to a file]"
        if expect_type:
            usage = f"Usage: {prefix}{message.command[0]} [custom prompt] [Reply to a {expect_type}]"
        return await message.edit_text(usage)

    if not _valid_file(reply, file_type=expect_type):
        type_text = expect_type if expect_type else "supported"
        return await message.edit_text(f"Invalid {type_text} file. Please try again.")

    await message.edit_text(status_msg)

    file_path = await reply.download()
    if not file_path or not os.path.exists(file_path):
        return await message.edit_text("Failed to process the file. Try again.")

    try:
        input_data = await prepare_input_data(reply, file_path, prompt)

        model = genai.GenerativeModel(
            MODEL_NAME, generation_config=COOK_GEN_CONFIG if cook_mode else None
        )

        for _ in range(3):
            try:
                response = model.generate_content(input_data)

                if not getattr(response, "candidates", []):
                    return await message.edit_text(
                        "Could not generate response."
                    )

                break

            except Exception as e:
                msg = str(e).lower()
                if "mimetype parameter" in msg and "not supported" in msg:
                    if expect_type is None:
                        return await message.edit_text("Invalid file type. Please try again.")
                    else:
                        raise
                if any(x in msg for x in ("403", "429", "permission", "quota")):
                    await asyncio.sleep(2)
                else:
                    raise
        else:
            raise e

        result_text = (f"Prompt: {prompt}\n" if show_prompt else "") + \
                      f"Answer: {getattr(response, 'text', '') or 'No content generated.'}"

        if len(result_text) > 4000:
            for i in range(0, len(result_text), 4000):
                await message.reply_text(result_text[i:i+4000], parse_mode=enums.ParseMode.MARKDOWN)
            await message.delete()
        else:
            await message.edit_text(result_text, parse_mode=enums.ParseMode.MARKDOWN)

    except ValueError as e:
        await message.edit_text(str(e))

    except Exception as e:
        await message.edit_text(f"Error: {format_exc(e)}")

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@Client.on_message(filters.command("getai", prefix) & filters.me)
async def getai(_, message):
    prompt = (
        message.text.split(maxsplit=1)[1]
        if len(message.command) > 1
        else "Get details of the image. Be accurate and write a short response."
    )
    await ai_process_handler(message, prompt, show_prompt=len(message.command) > 1,
                             expect_type="image", status_msg="Scanning...")

@Client.on_message(filters.command("aicook", prefix) & filters.me)
async def aicook(_, message):
    await ai_process_handler(
        message,
        "Identify the baked good in the image and provide an accurate recipe.",
        cook_mode=True,
        expect_type="image",
        status_msg="Cooking..."
    )

@Client.on_message(filters.command("aiseller", prefix) & filters.me)
async def aiseller(_, message):
    if len(message.command) > 1:
        target = message.text.split(maxsplit=1)[1]
        prompt = f"Generate a marketing description for the product.\nTarget Audience: {target}"
        await ai_process_handler(message, prompt, expect_type="image",
                                 status_msg="Generating description...")
    else:
        await message.edit_text(f"Usage: {prefix}aiseller [target audience] [Reply to an image]")

@Client.on_message(filters.command(["transcribe", "ts"], prefix) & filters.me)
async def transcribe(_, message):
    prompt = (
        message.text.split(maxsplit=1)[1]
        if len(message.command) > 1
        else "Transcribe it. Write only the transcription text."
    )
    await ai_process_handler(message, prompt, show_prompt=len(message.command) > 1,
                             expect_type="audio", status_msg="Transcribing...")

@Client.on_message(filters.command(["process", "pr"], prefix) & filters.me)
async def pr_command(_, message):
    args = message.text.split(maxsplit=1)
    show_prompt = len(args) > 1
    prompt = args[1] if show_prompt else "Shortly summarize the content and details of the file."
    await ai_process_handler(message, prompt, show_prompt=show_prompt)

modules_help["generative"] = {
    "getai [prompt] [reply image]": "Analyze an image using AI.",
    "aicook [reply image]": "Identify food and generate cooking instructions.",
    "aiseller [target audience] [reply image]": "Generate marketing text for products.",
    "transcribe [prompt] [reply audio/video]": "Transcribe or summarize audio/video.",
    "process [prompt] [reply any file]": "Process any file (image/audio/video/PDF/etc).",
}
