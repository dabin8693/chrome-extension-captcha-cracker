import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if __name__ == '__main__':
    # # 기존 그래프 모델 -> 프리즌 모델로 저장
    # # 모델 로드 및 빌드
    # model = tf.keras.models.load_model("capcha_cracker_final")
    #
    # # 모델을 함수로 변환
    # full_model = tf.function(lambda x: model(x))
    #
    # # 콘크리트 함수 생성
    # concrete_func = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    #
    # # 프리징: 변수를 상수로 변환
    # frozen_func = convert_variables_to_constants_v2(concrete_func)
    #
    # # 프리즌 그래프 저장
    # tf.io.write_graph(frozen_func.graph, "frozen_m", "frozen_graph.pb", as_text=False)
#///////////////////////////////////////////#
    # 저장된 프리즌 모델출력층 노드 이름 찾기
    # 프리즌 그래프 로드
    with tf.io.gfile.GFile("frozen_m/frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # 출력 노드 이름 확인
    for node in graph_def.node:
        print(node.name)
