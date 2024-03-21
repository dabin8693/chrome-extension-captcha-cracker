import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':

    model = tf.keras.models.load_model("capcha_cracker_model_j2")
    # tfjs에서 지원하지 않는 api는 제거 후 자체 제작
    # Preview Dataset
    img_list = glob('sample/test_numbers_2/*.jpg')
    # 단일 데이터 테스트
    x_val = img_list[61]

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
        # 3. Convert to float32 in [0, 1] range 정규화
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # shape=(50, 200, 1)
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 데이터 갯수를 담당하는 차원 늘리기
        img = tf.expand_dims(img,axis=0)
        # 7. Return a dict as our model is expecting two inputs
        return {'image': img}


    # Test Inference
    prediction_model = keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense2').output
    )
    # tfjs에서 쓰기 위해 label input layer제거, ctc loss output layer제거하고 저장
    prediction_model.save("capcha_cracker_final", save_format='tf')

    def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
        # ctc_decode 자체 제작
        # https://github1s.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_decoder.h 참조
        input_shape = tf.shape(y_pred)
        num_samples, num_steps = input_shape[0], input_shape[1]
        y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
        input_length = tf.cast(input_length, tf.int32)

        listvar = []
        for value in y_pred:
            listvar.append(tf.squeeze(value).numpy().argmax(axis=0).tolist())

        # 여기서 반복문 조건문 돌면서 뒷 숫자와 다른 숫자는 따로 string에 합쳐서 저장
        strvar = ""
        listvar = list(map(str, listvar))
        for i in range(0, len(listvar)-1):
            if (listvar[i] != listvar[i+1]) :
                strvar = strvar + listvar[i]
        strvar = strvar + listvar[-1]

        strvar = strvar.replace("10","")
        # int -> string 변경후 문자열 합치고 string으로 반환

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

    # ocr결과 디코딩
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :max_length
                  ]

        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
            output_text.append(res)
        return output_text


    preds = prediction_model.predict(encode_single_sample(x_val))

    # 결과 디코딩(ocr결과값)
    pred_texts = decode_batch_predictions(preds)

    print("값:", pred_texts)


    plot_model(prediction_model, show_shapes=True, to_file='model2.png')
    plt.show()

