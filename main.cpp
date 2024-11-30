#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>

#define SERVER_IP "127.0.0.1"  
#define SERVER_PORT 12345

struct Position {
    cv::Rect eye1;
    cv::Rect eye2;
    cv::Rect nose;
    bool is_set = false;
};

void send_position_to_server(const Position& position) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "opening socket err\n";
        return;
    }

    struct sockaddr_in server_address{};
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, SERVER_IP, &server_address.sin_addr) <= 0) {
        std::cerr << "IP err\n";
        close(sock);
        return;
    }

    if (connect(sock, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "connection err\n";
        close(sock);
        return;
    }

    std::string message = "Oko 1: " + std::to_string(position.eye1.x) + "," + std::to_string(position.eye1.y) + 
                          " Oko 2: " + std::to_string(position.eye2.x) + "," + std::to_string(position.eye2.y) + 
                          " Nos: " + std::to_string(position.nose.x) + "," + std::to_string(position.nose.y);

    ssize_t bytes_sent = send(sock, message.c_str(), message.length(), 0);
    if (bytes_sent < 0) {
        std::cerr << "sending data err\n";
    } else {
        std::cout << "Wysłano dane do serwera: " << message << "\n";
    }

    close(sock);
}

void print_position_difference(const Position& initial, const Position& current) {
    if (!initial.is_set) return;

    std::cout << "Zmiana pozycji względem początkowej:\n";
    std::cout << "Oko 1: dx=" << (current.eye1.x - initial.eye1.x)
              << ", dy=" << (current.eye1.y - initial.eye1.y) << "\n";
    std::cout << "Oko 2: dx=" << (current.eye2.x - initial.eye2.x)
              << ", dy=" << (current.eye2.y - initial.eye2.y) << "\n";
    std::cout << "Nos: dx=" << (current.nose.x - initial.nose.x)
              << ", dy=" << (current.nose.y - initial.nose.y) << "\n";
}

int main() {
    cv::CascadeClassifier face_cascade, eye_cascade, nose_cascade;
    if (!face_cascade.load("../haarcascade_frontalface_default.xml")) {
        std::cerr << "face cascade err" << std::endl;
        return -1;
    }
    if (!eye_cascade.load("../haarcascade_eye.xml")) {
        std::cerr << "eye_cascade err" << std::endl;
        return -1;
    }
    if (!nose_cascade.load("../haarcascade_mcs_nose.xml")) {
        std::cerr << "nose_cascade err" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "cam error" << std::endl;
        return -1;
    }

    Position initial_position, current_position;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat flipped_frame;
        cv::flip(frame, flipped_frame, 1);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(flipped_frame, faces, 1.1, 4, 0, cv::Size(120, 120));

        for (const auto& face : faces) {
            cv::Mat face_pos = flipped_frame(face);

            std::vector<cv::Rect> eyes;
            eye_cascade.detectMultiScale(face_pos, eyes, 1.1, 10, 0, cv::Size(15, 15));

            std::vector<cv::Rect> eye_rects;
            for (const auto& eye : eyes) {
                cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                eye_rects.push_back(eye_rect);
                cv::rectangle(flipped_frame, eye_rect, cv::Scalar(0, 255, 0), 2);
            }

            std::vector<cv::Rect> noses;
            nose_cascade.detectMultiScale(face_pos, noses, 1.1, 5, 0, cv::Size(30, 30));

            if (!noses.empty()) {
                current_position.nose = cv::Rect(face.x + noses[0].x, face.y + noses[0].y, noses[0].width, noses[0].height);
                cv::rectangle(flipped_frame, current_position.nose, cv::Scalar(0, 0, 255), 2);
            }

            if (eye_rects.size() >= 2) {
                current_position.eye1 = eye_rects[0];
                current_position.eye2 = eye_rects[1];
            }

            if (cv::waitKey(1) == 'p' && !initial_position.is_set) {
                initial_position = current_position;
                initial_position.is_set = true;
                std::cout << "Oko 1: " << initial_position.eye1 << "\n";
                std::cout << "Oko 2: " << initial_position.eye2 << "\n";
                std::cout << "Nos: " << initial_position.nose << "\n";
            }

            if (initial_position.is_set) {
                print_position_difference(initial_position, current_position);
            }

            if (cv::waitKey(1) == 's' && initial_position.is_set) {
                send_position_to_server(current_position);
            }
        }

        cv::imshow("Obraz", flipped_frame);
        if (cv::waitKey(10) == 27) break;  
    }

    return 0;
}
