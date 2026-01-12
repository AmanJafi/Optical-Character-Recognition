#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <allheaders.h>
#include <iostream>

using namespace cv;
using namespace std;


const int GRID  = 32;   
const int SCALE = 10;   


Mat canvas(GRID * SCALE, GRID * SCALE, CV_8UC1, Scalar(0));
bool drawing = false;


void drawPixel(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) drawing = true;
    if (event == EVENT_LBUTTONUP)   drawing = false;

    if (drawing) {
        int px = x / SCALE;
        int py = y / SCALE;

        if (px >= 0 && px < GRID && py >= 0 && py < GRID) {
            rectangle(
                canvas,
                Point(px * SCALE, py * SCALE),
                Point((px + 1) * SCALE, (py + 1) * SCALE),
                Scalar(255),
                FILLED
            );
        }
    }
}


void runOCR(Mat img) {

    resize(img, img, Size(128, 128));


    bitwise_not(img, img);


    GaussianBlur(img, img, Size(3, 3), 0);
    threshold(img, img, 0, 255, THRESH_BINARY | THRESH_OTSU);


    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
        cout << "Could not initialize Tesseract\n";
        return;
    }


    ocr.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);


    ocr.SetVariable(
        "tessedit_char_whitelist",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
    );


    ocr.SetVariable("load_system_dawg", "0");
    ocr.SetVariable("load_freq_dawg", "0");


    ocr.SetImage(
        img.data,
        img.cols,
        img.rows,
        1,
        img.step
    );

    char* text = ocr.GetUTF8Text();
    cout << "OCR Output: " << (text ? text : "") << endl;

    delete[] text;
    ocr.End();
}


int main() {
    namedWindow("Draw (P = Predict, C = Clear)", WINDOW_AUTOSIZE);
    setMouseCallback("Draw (P = Predict, C = Clear)", drawPixel);

    while (true) {
        imshow("Draw (P = Predict, C = Clear)", canvas);
        char key = waitKey(1);

        if (key == 27) break;       
        if (key == 'c') canvas.setTo(0);
        if (key == 'p') runOCR(canvas.clone());
    }

    return 0;
}

