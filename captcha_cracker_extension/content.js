window.addEventListener("load", function() {
    const url = window.location.hostname;
    const filePath = window.location.pathname;
    if ((url.indexOf("newtoki") != -1 )|| (url.indexOf("manatoki") != -1 )|| (url.indexOf("booktoki") != -1 )) {
        if(filePath.indexOf("bbs/captcha") != -1) {
            tf.setBackend('cpu').then(() => {
                capcha_processing(document.getElementsByClassName("captcha_img")[0].src);
            });
        }
    }
    
});




function capcha_processing(captcha_img_src) {
    const IMG = new Image();
    IMG.src = captcha_img_src;
    IMG.addEventListener('load', async function () {
        const finalImage = await image_preprocessing(IMG);
        modelURL = browser.runtime.getURL('/frozen_graph_m/model.json')
        // 파이어폭스 사용시 manifest파일의 content_scripts js부분에 tfjs4.17.0.js대신 tfjs4.17.0_feat.firefox.js사용
        const model = await tf.loadGraphModel(modelURL);
        //const model2 = await tf.loadGraphModel('chrome-extension://' + appID + '/captcha_cracker_web_final/model.json');

        //console.time('Model predict Time');  // 타이머 시작
        model.executeAsync(finalImage).then((prediction) => {
            const value = prediction.dataSync()
            //console.timeEnd('Model predict Time');  // 타이머 종료
            const captcha_number = ctc_greedy_decoder(value);
            document.getElementById("captcha_key").value = captcha_number
            document.getElementsByName("fcaptcha")[0].submit()
        });

    });

}



function image_preprocessing(IMG) {
    let image = tf.browser.fromPixels(IMG, 1);
    image = tf.image.resizeBilinear(image, [50, 200]).div(tf.scalar(255)).toFloat();
    image = tf.transpose(image, perm = [1, 0, 2]);
    image = tf.expandDims(image, axis = 0);
    return image;
}

function ctc_greedy_decoder(value){
    let index = 0;
    let list = [];
    let maxnumindex = 0;
    let maxnum = 0;
    for(const element of value) {
        if (index == 0) {
            maxnum = element
            maxnumindex = 0
        }
        if (maxnum<element) {
            maxnum = element
            maxnumindex = index
        }
        if (index==10) {
            list.push(maxnumindex)
            index = 0;
            maxnumindex = 0;
        }
        else
        {
            index++
        }
    }
    let list2 = [];
    let strNum = "";
    let captch_num = "";
    for (let i = 0; i<49; i++) {
        if (list[i] != list[i + 1]) {
            list2.push(list[i])
        }
    }
    list2.push(list[list.length-1])
    strNum = list2.join('')
    captch_num = strNum.replaceAll(10, '');
    return captch_num
}
