window.addEventListener("load", function () {
    const url = window.location.hostname;
    const filePath = window.location.pathname;

    //bbs/captcha
    if ((url.indexOf("newtoki") != -1) || (url.indexOf("manatoki") != -1) || (url.indexOf("booktoki") != -1)) {
        if (filePath.indexOf("bbs/captcha") != -1) {
            capcha_processing(document.getElementsByClassName("captcha_img")[0].src);

        }
    }
});

let port;

// 페이지가 다시 활성화되면 포트를 다시 연결
window.addEventListener('pageshow', () => {
    if (!port) {
        connectPort();
    }
});

function capcha_processing(captcha_img_src) {
    const IMG = new Image();
    IMG.src = captcha_img_src;
    IMG.onload = function () {
        let canvas = document.createElement('canvas');
        canvas.width = IMG.width;
        canvas.height = IMG.height;
        let ctx = canvas.getContext('2d');
        ctx.drawImage(IMG, 0, 0);
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // 포트 처음 연결
        connectPort();

        port.postMessage({
            action: 'processImage', imageData: {
                data: Array.from(imageData.data),  // Uint8ClampedArray의 buffer를 전송
                width: imageData.width,
                height: imageData.height
            }
        });

    }
}

function connectPort() {
    port = chrome.runtime.connect({ name: "captchaPort" });

    port.onMessage.addListener((response) => {
        if (response.result) {
            console.log('캡차 번호:', response.result);
            document.getElementById("captcha_key").value = response.result;
            document.getElementsByName("fcaptcha")[0].submit();
        } else if (response.error) {
            console.error('오류 발생:', response.error);
        }
    });

    port.onDisconnect.addListener(() => {
        console.log("포트가 닫혔습니다. 페이지가 BFCache에 이동했을 수 있습니다.");
        port = null;
    });
}




