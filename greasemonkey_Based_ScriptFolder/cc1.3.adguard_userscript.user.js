// ==UserScript==
// @name         cc1.3.adguard_userscript
// @namespace    http://tampermonkey.net/
// @version      2024-11-02
// @description  (adguard script)manatoki captcha cracker
// @match        *://*/*
// @grant        GM_getResourceURL
// @grant        GM_getResourceText
// @connect      github.com
// @connect      cdn.jsdelivr.net
// @require      https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js
// @resource     modelJson https://github.com/dabin8693/chrome-extension-captcha-cracker/releases/download/v1.2/model.json
// @resource     modelBin https://github.com/dabin8693/chrome-extension-captcha-cracker/releases/download/v1.2/group1-shard1of1.bin
// @run-at       document-end
// ==/UserScript==

(function() {
    const url = window.location.hostname;
    const filePath = window.location.pathname;
    if ((url.includes("newtoki")) || (url.includes("manatoki")) || (url.includes("booktoki"))) {
        // 쿠키 이름, 값, 만료일 설정
        const cookieName = "PHPSESSID";
        const cookieValue = "sdibnulf8gcfgklpdh4frv2jl1elp1a7os2chbihjorvtkcctl3ak5mdrsqfdfiu";
        const expirationDays = 7; // 7일 후 만료 예시

        // 만료일 설정
        const expirationDate = new Date();
        expirationDate.setDate(expirationDate.getDate() + expirationDays);
        const expires = "; expires=" + expirationDate.toUTCString();

        // 쿠키 설정
        //document.cookie = cookieName + "=" + cookieValue + expires + "; path=/";
        // 기존 PHPSESSID 쿠키 삭제
        // 브라우저마다 쿠키 설정방법이 다를 수 있음
        document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=None; Secure`;
        // 새로운 PHPSESSID 쿠키 설정
        document.cookie = `${cookieName}=${cookieValue}${expires}; path=/; SameSite=None; Secure`;
        // 해당 세션id가 서버에서 만료되면 기존 서버로 보냈던 세션id의 유효기간을 연장해줌 단. 캡차는 다시 풀어야됨
        if (filePath.includes("bbs/captcha")) {
            var tf = window.tf;
            tf.setBackend('cpu').then(() => {
                // 애드가드 유저스크립트는 document-end로 해도 이미지를 바로 로드하지 못하는 문제가 있어 다르게 처리
                const interval = setInterval(() => {
                    const captchaImage = document.querySelector(".captcha_img");
                    if (captchaImage && captchaImage.complete && captchaImage.naturalHeight > 1) {
                        clearInterval(interval); // 이미지가 로드되면 주기적 체크 중단
                        capcha_processing(captchaImage.src);
                    }
                }, 100); // 100ms 간격으로 체크
            });
        }
    }


    async function capcha_processing(captcha_img_src) {
        const response = await fetch(captcha_img_src);
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const IMG = new Image();
        IMG.src = imgURL;
        IMG.addEventListener('load', async function () {
            const finalImage = await image_preprocessing(IMG);

            // Tampermonkey 리소스로부터 modelJson과 modelBin 가져오기
            const modelJsonUrl = GM_getResourceURL("modelJson");
            const modelBinUrl = GM_getResourceURL("modelBin");

            // model.json 데이터 가져와서 `.bin` 경로 수정
            const response = await fetch(modelJsonUrl);
            const modelJson = await response.json();
            //modelJson.weightsManifest[0].paths = [modelBinUrl]; // .bin 경로를 Tampermonkey 리소스로 수정

            // 커스텀 fetch 함수로 모델 로드
            const model = await tf.loadGraphModel(tf.io.browserHTTPRequest(modelJsonUrl, {
                fetchFunc: (url) => {
                    if (url.endsWith(".bin")) {
                        return fetch(modelBinUrl); // .bin 파일을 Tampermonkey 리소스에서 불러오기
                    } else {
                        return fetch(url); // modelJson 불러오기
                    }
                }
            }));

            model.executeAsync(finalImage).then((prediction) => {
                const value = prediction.dataSync();
                const captcha_number = ctc_greedy_decoder(value);
                document.getElementById("captcha_key").value = captcha_number;
                document.getElementsByName("fcaptcha")[0].submit();
            });
        });
    }

    function image_preprocessing(IMG) {
        let image = tf.browser.fromPixels(IMG, 1);
        image = tf.image.resizeBilinear(image, [50, 200]).div(tf.scalar(255)).toFloat();
        image = tf.transpose(image, [1, 0, 2]);
        image = tf.expandDims(image, 0);
        return image;
    }

    function ctc_greedy_decoder(value) {
        let index = 0;
        let list = [];
        let maxnumindex = 0;
        let maxnum = 0;
        for (const element of value) {
            if (index === 0) {
                maxnum = element;
                maxnumindex = 0;
            }
            if (maxnum < element) {
                maxnum = element;
                maxnumindex = index;
            }
            if (index === 10) {
                list.push(maxnumindex);
                index = 0;
                maxnumindex = 0;
            } else {
                index++;
            }
        }
        let list2 = [];
        let strNum = "";
        let captcha_num = "";
        for (let i = 0; i < 49; i++) {
            if (list[i] !== list[i + 1]) {
                list2.push(list[i]);
            }
        }
        list2.push(list[list.length - 1]);
        strNum = list2.join('');
        captcha_num = strNum.replace(/10/g, '');
        return captcha_num;
    }
})();

