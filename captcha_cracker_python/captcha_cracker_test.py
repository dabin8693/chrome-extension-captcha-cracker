import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':

    model = tf.keras.models.load_model("capcha_cracker_model_j2")
    # Preview Dataset
    img_list = glob('sample/test_num/*.jpg')

    x_val = img_list
    y_val = []
    for label in x_val:
        y_val.append(os.path.splitext(os.path.basename(x_val[0]))[0])

    # Preprocessing

    max_length = 4

    characters2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Encode Labels
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=characters2, num_oov_indices=0, mask_token=None
    )

    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True
    )

    # Create Data Generator
    img_width = 200
    img_height = 50


    def encode_single_sample(img_path):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        #label = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
        # 7. Return a dict as our model is expecting two inputs
        return {'image': img}


    #preview = encode_single_sample(imgs[0], labels[0])

    batch_size = len(img_list)
    encode_single_sample(x_val[0])
    validation_dataset = tf.data.Dataset.from_tensor_slices(x_val).map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Test Inference
    prediction_model = keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense2').output
    )
    # ocr결과 디코딩
    def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
        input_shape = tf.shape(y_pred)
        num_samples, num_steps = input_shape[0], input_shape[1]
        y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
        input_length = tf.cast(input_length, tf.int32)

        if greedy:
            (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
                inputs=y_pred, sequence_length=input_length, merge_repeated=True
            )
        else:
            (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=top_paths,
                merge_repeated=False
            )

        decoded_dense = []
        for st in decoded:
            st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
            decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
        return (decoded_dense, log_prob)

    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # beam_search 사용시 merge_repeated true로 사용
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :max_length
                  ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
            output_text.append(res)
        return output_text

    for batch in validation_dataset.take(1):
        batch_images = batch['image']
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        _, axes = plt.subplots(8, 4, figsize=(16, 12))
        index = 0
        for img, text in zip(batch_images, pred_texts):
            img = img.numpy().squeeze()
            img = img.T
            row = index%8
            col = index//8
            axes[row][col].imshow(img, cmap='gray')
            axes[row][col].set_title(text)
            axes[row][col].set_axis_off()
            index += 1
            if index > 31 :
                index = 0
                plt.show()
                _, axes = plt.subplots(8, 4, figsize=(16, 12))

    plt.show()
