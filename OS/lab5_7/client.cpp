#include <iostream>
#include <unistd.h>
#include <string>
#include <sstream>
#include <exception>
#include <signal.h>
#include "zmq.hpp"

const int PORTBASE  = 5050;

bool sendMessage(zmq::socket_t& socket, const std::string& message) {
    zmq::message_t zmqMessage(message.c_str(), message.size());
    zmq::send_result_t result = socket.send(zmqMessage, zmq::send_flags::none);
    return result.has_value();
}

std::string recieveMessage(zmq::socket_t &socket) {
    zmq::message_t message;
    bool ok = false;
    try {
        zmq::recv_result_t result = socket.recv(message, zmq::recv_flags::none);
        ok = result.has_value();
    }
    catch (...) {
        ok = false;
    }
    
    std::string received_message(static_cast<char*>(message.data()), message.size());
    if (received_message.empty() || !ok)
        return "Error: Node is not available";
    
    return received_message;
}

void createNode(int id, int port) {
    char* arg0 = strdup("./client");
    char* arg1 = strdup((std::to_string(id)).c_str());
    char* arg2 = strdup((std::to_string(port)).c_str());
    char* args[] = {arg0, arg1, arg2, NULL};
    execv("./client", args);
}

std::string getPortName(const int port) {
    return "tcp://127.0.0.1:" + std::to_string(port);
}

void create(zmq::socket_t& parentsocket, zmq::socket_t& socket, int& create_id, int& id, int& pid) {
    if (pid == -1) {
        sendMessage(parentsocket, "Error: Cannot fork");
        pid = 0;
    } 
    else if (pid == 0) 
        createNode(create_id,PORTBASE + create_id);

    else {
        id = create_id;
        sendMessage(socket, "pid");
        sendMessage(parentsocket, recieveMessage(socket));
    }
}

void kill(zmq::socket_t& parentsocket, zmq::socket_t& socket,  int& delete_id, int& id, int& pid, std::string& request_string) {
    if (id == 0)
        sendMessage(parentsocket, "Error: Not found");

    else if (id == delete_id) {
        sendMessage(socket, "kill_children");
        recieveMessage(socket);
        kill(pid,SIGTERM);
        kill(pid,SIGKILL);
        id = 0;
        pid = 0;
        sendMessage(parentsocket, "Ok");
    } 
    else {
        sendMessage(socket, request_string);
        sendMessage(parentsocket, recieveMessage(socket));
    }
}

void rl_exec(zmq::socket_t& parentsocket, zmq::socket_t& socket,  int& id, int& pid, std::string& request_string) {
    if (pid == 0) {
        std::string recieveMessage = "Error:" + std::to_string(id);
        recieveMessage += ": Not found";
        sendMessage(parentsocket, recieveMessage);
    } 
    else {
        sendMessage(socket, request_string);
        sendMessage(parentsocket, recieveMessage(socket));
    }
}

void exec(std::istringstream& request_stream, zmq::socket_t& parentsocket, zmq::socket_t& left_socket, 
            zmq::socket_t& right_socket, int& left_pid, int& right_pid, int& id, std::string& request_string) {
    std::string size_str;
    std::string number;
    int exec_id;
    request_stream >> exec_id;
    if (exec_id == id) {
        request_stream >> size_str;
        int size=stoi(size_str);
        int sum=0;
        for (size_t i = 0; i < size; i++) {
            request_stream >> number;
            sum+=stoi(number);
        }
        
        std::string recieveMessage = std::to_string(sum);
        sendMessage(parentsocket, recieveMessage);
    } 
    else if (exec_id < id) {
        rl_exec(parentsocket, left_socket, exec_id,
                left_pid, request_string);
    } 
    else {
        rl_exec(parentsocket, right_socket, exec_id,
                right_pid, request_string);
    }
}


void pingall(zmq::socket_t& parentsocket, int& id, zmq::socket_t& left_socket, zmq::socket_t& right_socket,int& left_pid, int& right_pid) {
    std::ostringstream res;
    std::string left_res;
    std::string right_res;        
    res << std::to_string(id);
    if (left_pid != 0) {
        sendMessage(left_socket, "pingall");
        left_res = recieveMessage(left_socket);
    }
    if (right_pid != 0) {
        sendMessage(right_socket, "pingall");
        right_res = recieveMessage(right_socket);
    }
    if (!left_res.empty() && left_res.substr(0, std::min<int>(left_res.size(),5) ) != "Error") {
        res << " " << left_res;        
    }
    if ((!right_res.empty()) && (right_res.substr(0, std::min<int>(right_res.size(),5) ) != "Error")) {
        res << " "<< right_res;
    }
    sendMessage(parentsocket, res.str());
}

void kill_children(zmq::socket_t& parentsocket, zmq::socket_t& left_socket, zmq::socket_t& right_socket, int& left_pid, int& right_pid) {
    if (left_pid == 0 && right_pid == 0)
        sendMessage(parentsocket, "Ok");

    else {
        if (left_pid != 0) {
            sendMessage(left_socket, "kill_children");
            recieveMessage(left_socket);
            kill(left_pid,SIGTERM);
            kill(left_pid,SIGKILL);
        }
        if (right_pid != 0) {
            sendMessage(right_socket, "kill_children");
            recieveMessage(right_socket);
            kill(right_pid,SIGTERM);
            kill(right_pid,SIGKILL);
        }
        sendMessage(parentsocket, "Ok");
    }
}

int main(int argc, char** argv) {
    int id = std::stoi(argv[1]);
    int parent_port = std::stoi(argv[2]);
    zmq::context_t context(3);
    zmq::socket_t parentsocket(context, ZMQ_REP);
    parentsocket.connect(getPortName(parent_port));
    int left_pid = 0;
    int right_pid = 0;
    int left_id = 0;
    int right_id = 0;
    zmq::socket_t left_socket(context, ZMQ_REQ);
    zmq::socket_t right_socket(context, ZMQ_REQ);
    while(true) {
        std::string request_string = recieveMessage(parentsocket);
        std::istringstream request_stream(request_string);
        std::string request;
        request_stream >> request;
        if (request == "id") {
            std::string parent_string = "Ok:" + std::to_string(id);
            sendMessage(parentsocket, parent_string);
        } 
        else if (request == "pid") {
            std::string parent_string = "Ok:" + std::to_string(getpid());
            sendMessage(parentsocket, parent_string);
        } 
        else if (request == "create") {
            int create_id;
            request_stream >> create_id;
            if (create_id == id) {
                std::string message_string = "Error: Already exists";
                sendMessage(parentsocket, message_string);
            } 
            else if (create_id < id) {
                if (left_pid == 0) {
                    left_socket.bind(getPortName(PORTBASE + create_id));
                    left_pid = fork();
                    create(parentsocket, left_socket, create_id, left_id, left_pid);
                } 
                else {
                    sendMessage(left_socket, request_string);
                    sendMessage(parentsocket, recieveMessage(left_socket));
                }
            } 
            else {
                if (right_pid == 0) {
                    right_socket.bind(getPortName(PORTBASE + create_id));
                    right_pid = fork();
                    create(parentsocket, right_socket, create_id, right_id, right_pid);
                } 
                else {
                    sendMessage(right_socket, request_string);
                    sendMessage(parentsocket, recieveMessage(right_socket));
                }
            }
        } 
        else if (request == "kill") {
            int delete_id;
            request_stream >> delete_id;
            if (delete_id < id)
                kill(parentsocket, left_socket, delete_id, left_id, left_pid, request_string);
 
            else
                kill(parentsocket, right_socket, delete_id, right_id, right_pid, request_string);
        } 
        else if (request == "exec")
            exec(request_stream, parentsocket, left_socket, right_socket, left_pid, right_pid, id, request_string);

        else if (request == "pingall")
            pingall(parentsocket, id, left_socket, right_socket, left_pid, right_pid);

        else if (request == "kill_children")
            kill_children(parentsocket, left_socket, right_socket, left_pid, right_pid); 

        if (parent_port == 0) 
            break;
    }

    return 0;
}