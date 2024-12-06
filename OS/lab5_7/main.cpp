#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <signal.h>
#include "zmq.hpp"
#include "tree.h"

const int PORTBASE = 5050;
Tree T;
std::string request;
int childProcessId = 0;
int childId = 0;

zmq::context_t context(1); // 1 - один I/O поток
zmq::socket_t mainSocket(context, ZMQ_REQ); // Request-reply

bool sendMessage(zmq::socket_t &socket, const std::string &message) {
    zmq::message_t zmqMessage(message.c_str(), message.size());
    zmq::send_result_t result = socket.send(zmqMessage, zmq::send_flags::none);

    return result.has_value(); // Результат отправки (T or F)
}

std::string recieveMessage(zmq::socket_t &socket) {
    zmq::message_t message;
    bool ok = false;
    try { // В случае успеха вернём сообщение
        zmq::recv_result_t result = socket.recv(message, zmq::recv_flags::none);
        ok = result.has_value();
    }
    catch (...) {
        ok = false;
    }

    std::string receivedMessage(static_cast<char *>(message.data()), message.size());
    if (receivedMessage.empty() || !ok)
        return "Error: Node is not available";

    return receivedMessage;
}

void createNode(int id, int port) {
    char *arg0 = strdup("./client");
    char *arg1 = strdup((std::to_string(id)).c_str());
    char *arg2 = strdup((std::to_string(port)).c_str());
    char *args[] = {arg0, arg1, arg2, NULL};
    execv("./client", args);
}

// Преобразуем номер порта в строку нужного формата для ZMQ
std::string getPortName(const int port) {
    return "tcp://127.0.0.1:" + std::to_string(port);
}

bool isNumber(std::string value) {
    try {
        int check = std::stoi(value);
        return true;
    }
    catch (std::exception &e) {
        std::cout << "Error: " << e.what() << "\n";
        return false;
    }
}

void funcCreate() {
    size_t nodeId = 0;
    std::string lineInput = "";
    std::string result = "";
    std::cin >> lineInput;
    if (!isNumber(lineInput))
        return;

    nodeId = stoi(lineInput);
    if (childProcessId == 0) {
        mainSocket.bind(getPortName(PORTBASE + nodeId)); // Привязываем сокет к порту
        childProcessId = fork();
        if (childProcessId == -1) {
            std::cout << "Unable to create first worker node\n";
            childProcessId = 0;
            exit(1);
        }
        else if (childProcessId == 0)
            createNode(nodeId, PORTBASE + nodeId);

        else {
            childId = nodeId;
            sendMessage(mainSocket, "pid");
            result = recieveMessage(mainSocket);
        }
    }
    else { // Если дочерний процеес существует, отправляем сообщение о создании нового узла
        std::string mеssage_send = "create " + std::to_string(nodeId);
        sendMessage(mainSocket, mеssage_send);
        result = recieveMessage(mainSocket);
    }
    if (result.substr(0, 2) == "Ok")
        T.push(nodeId);

    std::cout << result << "\n";
}

void funcKill() {
    int nodeId = 0;
    std::string lineInput = "";
    std::cin >> lineInput;
    if (!isNumber(lineInput))
        return;

    nodeId = stoi(lineInput);
    if (childProcessId == 0) {
        std::cout << "Error: Not found\n";
        return;
    }
    if (nodeId == childId) { // Если дочерний процесс соответсвует узлу
        kill(childProcessId, SIGTERM);
        kill(childProcessId, SIGKILL);
        childId = 0;
        childProcessId = 0;
        T.kill(nodeId);
        std::cout << "Ok\n";
        return;
    }
    // Если не соответсвтует
    std::string message = "kill " + std::to_string(nodeId);
    sendMessage(mainSocket, message);
    std::string recieved_message = recieveMessage(mainSocket);
    if (recieved_message.substr(0, std::min<int>(recieved_message.size(), 2)) == "Ok")
        T.kill(nodeId);

    std::cout << recieved_message << "\n";
}

void funcExec() {
    std::string stringID = "";
    std::string amountNumbers = "";
    std::string number = "";

    int id = 0;
    std::cin >> stringID >> amountNumbers;
    if (!isNumber(stringID))
        return;

    if (!isNumber(amountNumbers))
        return;

    std::vector<std::string> inputNumbers;
    for (size_t i = 0; i < stoi(amountNumbers); i++) {
        std::cin >> number;
        inputNumbers.push_back(number);
    }

    id = stoi(stringID);

    std::string messageLine = "exec " + std::to_string(id) + " " + amountNumbers;
    for (size_t i = 0; i < stoi(amountNumbers); i++)
        messageLine += (" " + (inputNumbers[i]));

    sendMessage(mainSocket, messageLine);
    std::string result = recieveMessage(mainSocket);
    std::cout << result << "\n";
}

int main() {
    std::cout << "requests:\n";
    std::cout << "create id\n";
    std::cout << "exec id amount_num num1 num2...num_n\n";
    std::cout << "kill id\n";
    std::cout << "pingall\n";
    std::cout << "exit\n" << std::endl;

    while (1) {
        std::cin >> request;
        if (request == "create")
            funcCreate();

        else if (request == "kill")
            funcKill();

        else if (request == "exec")
            funcExec();

        else if (request == "pingall") {
            sendMessage(mainSocket, "pingall");
            std::string recieved = recieveMessage(mainSocket);
            std::istringstream is;
            if (recieved.substr(0, std::min<int>(recieved.size(), 5)) == "Error")
                is = std::istringstream("");

            else
                is = std::istringstream(recieved);

            std::set<int> recieved_T;
            int recievedId;
            while (is >> recievedId)
                recieved_T.insert(recievedId);

            std::vector<int> from_tree = T.get_nodes();
            auto part_it = partition(from_tree.begin(), from_tree.end(), [&recieved_T](int a) { return recieved_T.count(a) == 0; });
            if (part_it == from_tree.begin())
                std::cout << "Ok:-1\n";

            else {
                std::cout << "Ok:";
                for (auto it = from_tree.begin(); it != part_it; ++it)
                    std::cout << *it << " ";

                std::cout << "\n";
            }
        }
        else if (request == "exit") {
            int n = system("kill all client");
            break;
        }
    }

    return 0;
}