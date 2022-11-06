def execute():
    import os
    import tensorflow.keras
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
    from tensorflow.keras.models import Model
    import time

    model = tensorflow.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

    import tensorflow
    import numpy as np
    def load_image(path):
        img = tensorflow.keras.utils.load_img(path, target_size=model.input_shape[1:3])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    feat_extractor.summary()

    import random
    from matplotlib import pyplot as plt
    from scipy.spatial import distance

    def get_closest_images(query_image_idx, num_results=5):
        distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
        print(distances)
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results + 1]
        print(idx_closest)
        return idx_closest

    def get_concatenated_images(indexes, thumb_height):
        thumbs = []
        for idx in indexes:
            img = image.load_img(images[idx])
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        return concat_image

    from sklearn.decomposition import PCA
    import pickle as pk

    with open('content/features_images.p', 'rb') as pickle_file:
        images, pca_features, pca = pk.load(pickle_file)
        # do a query on a random image
        # query_image_idx = int(len(images) * random.random())

        imagetime = []
        for i in os.listdir("static/files"):
            imagetime.append(os.path.getmtime("static/files/"+str(i)))
        imagetime.sort()

        for i in range(len(imagetime)):
            if imagetime[-1] == os.path.getmtime("static/files/"+str(os.listdir("static/files")[i])):
                userimage = "static/files/"+str(os.listdir("static/files")[i])
            else: continue


        new_image, x = load_image(userimage)

        new_features = feat_extractor.predict(x)
        new_pca_features = pca.transform(new_features)[0]
        distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
        results_image = get_concatenated_images(idx_closest, 200)

        # idx_closest = get_closest_images(query_image_idx)
        # print(idx_closest)
        # query_image = get_concatenated_images([query_image_idx], 300)
        # results_image = get_concatenated_images(idx_closest, 200)

        # display the query image
        #plt.figure(figsize=(5, 5))
        #plt.imshow(new_image)
        #plt.title("query image (%d)")
        # plt.title("query image (%d)" % query_image_idx)

        # display the resulting images
        #plt.figure(figsize=(16, 12))
        #plt.imshow(results_image)
        #plt.title("result images")

        import pandas as pd
        df = pd.read_csv('content/HAM10000_metadata (1)')
        lesion = []

        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'dermatofibroma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        from PIL import Image
        import base64
        from io import BytesIO


        def encode_img(img):
            pil_img = Image.fromarray(img)
            buff = BytesIO()
            pil_img.save(buff, format="JPEG")
            results_image = base64.b64encode(buff.getvalue()).decode("utf-8")
            return results_image


        for i in idx_closest:
            x = df['dx'].loc[df.index[i]]
            lesion.append(lesion_type_dict[x])

        def encode_img1(image, im_type="JPEG"):
            buffered = BytesIO()
            image.save(buffered, format=im_type)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str


    new_image = encode_img1(new_image)
    results_image = encode_img(results_image)


    main_result = [lesion, new_image, results_image]
    return main_result

