from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mls_tourism/deployment",     # the local folder containing your file
    repo_id="v-vasanth2009/tourism-package-prediction-28032026",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
