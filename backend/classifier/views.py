import base64
import io

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

from .inference import predict


@csrf_exempt
def predict_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file under the 'image' field."}, status=405)

    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "No image file provided."}, status=400)

    try:
        pil_image = Image.open(image_file)

        preview_buf = io.BytesIO()
        pil_image.convert("RGB").save(preview_buf, format="PNG")
        preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("ascii")

        result = predict(pil_image)
        result["preview"] = f"data:image/png;base64,{preview_b64}"
        return JsonResponse(result)

    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)