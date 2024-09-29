importScripts('tfjs4.17.0.js')
const appID = chrome.runtime.id
let model;

async function loadModel() {
    model = await tf.loadGraphModel('chrome-extension://' + appID + '/captcha_cracker_web_final/model.json');
    //const model = await tf.loadGraphModel('chrome-extension://mchcbjjljbnddeabcnkhdlpppgfhgobk/captcha_cracker_web_final/model.json');
}
// 서비스 워커가 시작될 때 모델을 비동기로 로드
loadModel();

chrome.runtime.onConnect.addListener((port) => {
    port.onMessage.addListener(async (message) => {
        if (message.action === "processImage" && model) {
            try {
                const imgData = message.imageData;
                const imageDataArray = new Uint8ClampedArray(imgData.data);
                const reconstructedImageData = new ImageData(imageDataArray, imgData.width, imgData.height);
                const finalImage = image_preprocessing(reconstructedImageData);

                // 모델 실행 및 예측
                let value = await model.executeAsync(finalImage);
                value = await value.array();

                // ctc_greedy_decoder 실행
                let captcha_number = ctc_greedy_decoder(value);

                // 포트를 통해 결과 전송
                port.postMessage({ result: captcha_number });
            } catch (error) {
                console.error('오류 발생:', error);
                port.postMessage({ error: '처리 중 오류 발생' });
            }
        }
    });
});


function image_preprocessing(IMG) {
    let image = tf.browser.fromPixels(IMG, 1);
    image = tf.image.resizeBilinear(image, [50, 200]).div(tf.scalar(255)).toFloat();
    //255로 나눠 [0, 1] 정규화
    image = tf.transpose(image, perm = [1, 0, 2]);
    image = tf.expandDims(image, axis = 0);
    //1,200,50,1
    return image;
}


function ctc_greedy_decoder(value) {
    // 배열 평면화
    const flattened = value.flat(2);  // 2단계까지 평면화, 즉 50x11 배열을 1차원 배열로 변환

    let index = 0;
    let maxnumindex = 0;
    let maxnum = 0;
    let list = [];

    for (const element of flattened) {

        if (index === 0) {
            maxnum = element;
            maxnumindex = 0;
        }
        if (maxnum < element) {
            maxnum = element;
            maxnumindex = index;
        }
        if (index === 10) {
            console.log("index:", maxnumindex);
            list.push(maxnumindex);
            index = 0;
            maxnumindex = 0;
        } else {
            index++;
        }
    }
    console.log('list는:', list);
    let list2 = [];
    let strNum = "";
    let captch_num = "";
    for (let i = 0; i < 49; i++) {
        if (list[i] != list[i + 1]) {
            list2.push(list[i])
        }
    }
    console.log('list2는:', list2);
    list2.push(list[list.length - 1])
    strNum = list2.join('')
    captch_num = strNum.replaceAll(10, '');
    return captch_num
}