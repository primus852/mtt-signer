import requests


def download_file_from_google_drive(model_name, destination):
    url = "https://docs.google.com/uc?export=download"

    file_id = None
    if model_name == 'ssd_mobilenet_v2_coco':
        file_id = '1z-4Twst0HE-W3yVBX-ZIjSbwo7p6sSLH'
    elif model_name == 'faster_rcnn_resnet50_coco':
        file_id = '1usVQwrOMYpH88Dv3ulUZn1zzOIkbxB9H'
    else:
        print('COULD NOT DOWNLOAD MODEL %s' % model_name)
        exit()

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, '{}/frozen_inference_graph.pb'.format(destination))


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    chunk_size = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
