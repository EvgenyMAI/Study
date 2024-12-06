#include <iostream> // g++ -o dag dag.cpp -ljsoncpp -pthread
#include <fstream>
#include <jsoncpp/json/json.h>
#include <set>
#include <functional>
#include <vector>
#include <string>
#include <sys/wait.h>
#include <queue>
#include <unordered_set>
#include "unistd.h"
#include "sys/wait.h"
#include "stdio.h"
#include "stdlib.h"
#include "ctype.h"

void doingJob(const std::string& job, const std::unordered_map<std::string, std::vector<Json::String>>& dependencies, 
std::unordered_set<std::string>& visitedJobs) {
    if (visitedJobs.size() == dependencies.size())
        return;
    
    if (visitedJobs.count(job) == 0)
        visitedJobs.insert(job);
    
    const std::vector<Json::String>& currentDependencies = dependencies.at(job);

    for (const auto& dependency : currentDependencies) {
        if (visitedJobs.count(dependency) == 0) {
            visitedJobs.insert(dependency);
            if (visitedJobs.size() == dependencies.size()) {
                std::cout << "Выполняется работа: " << dependency << std::endl;
                sleep(1);
                std::cout << "Работа завершена: " << dependency << std::endl;
            } else
                doingJob(dependency, dependencies, visitedJobs);
        }
    }

    std::cout << "Выполняется работа: " << job << std::endl;
    sleep(1);
    std::cout << "Работа завершена: " << job << std::endl;
}

bool hasCycle(const Json::Value& jobs, const std::string& currentJob, std::set<std::string>& visited, std::set<std::string>& recursionStack) {
    visited.insert(currentJob);
    recursionStack.insert(currentJob);

    const Json::Value& dependencies = jobs[currentJob]["dependencies"];
    for (const auto& dependency : dependencies) {
        const std::string& dependencyJob = dependency.asString();

        if (recursionStack.count(dependencyJob))
            return true;

        if (!visited.count(dependencyJob) && hasCycle(jobs, dependencyJob, visited, recursionStack))
            return true;
    }

    recursionStack.erase(currentJob);
    return false;
}

// Ходим по всем узлам и проверяем, есть ли цикл
bool isValidDAG(const Json::Value& jobs) {
    std::set<std::string> visited;
    std::set<std::string> recursionStack;

    for (const auto& job : jobs.getMemberNames()) {
        if (!visited.count(job) && hasCycle(jobs, job, visited, recursionStack))
            return false;
    }

    return true;
}

bool isOneComponent(const Json::Value& jobs) {
    std::unordered_map<std::string, std::vector<std::string>> adjacencyList;
    std::set<std::string> visited;

    for (const auto& job : jobs.getMemberNames()) {
        const Json::Value& dependencies = jobs[job]["dependencies"];
        for (const auto& dependency : dependencies) {
            adjacencyList[dependency.asString()].push_back(job);
            adjacencyList[job].push_back(dependency.asString());
        }
    }

    int componentCount = 0;

    // Обходим граф и считаем количество компонент связности
    for (const auto& job : jobs.getMemberNames()) {
        if (visited.count(job) == 0) {
            visited.insert(job);

            std::set<std::string> component;
            std::queue<std::string> q;

            q.push(job);

            while (!q.empty()) {
                std::string currentJob = q.front();
                q.pop();
                component.insert(currentJob);

                for (const auto& neighbor : adjacencyList[currentJob]) {
                    if (visited.count(neighbor) == 0) {
                        q.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }

            componentCount++;

            if (componentCount > 1)
                return false;
        }
    }

    return true;
}

bool hasStartAndEndJobs(const Json::Value& jobs) {
    std::set<std::string> startJobs;
    std::set<std::string> endJobs;

    for (const auto& job : jobs.getMemberNames()) {
        const Json::Value& dependencies = jobs[job]["dependencies"];
        if (dependencies.empty()) {
            startJobs.insert(job);
        }

        for (const auto& dependency : dependencies) {
            endJobs.erase(dependency.asString());
        }

        endJobs.insert(job);
    }

    return !startJobs.empty() && !endJobs.empty();
}

int main() {
    std::ifstream file("test.json");
    Json::Value data;
    file >> data;
    Json::Value jobb = data["jobs"];

    if (!isValidDAG(jobb))
        throw std::runtime_error("DAG содержит цикл");

    if (!isOneComponent(jobb))
        throw std::runtime_error("DAG имеет больше одной компоненты связности");

    if (!hasStartAndEndJobs(jobb))
        throw std::runtime_error("DAG не имеет начального и конечного job'a");

    std::cout << "DAG валидный, приступаем к выполнению job's..." << std::endl;
    sleep(1);

    std::vector<std::string> jobs;
    for (const auto& job : jobb.getMemberNames())
        jobs.push_back(job);

    std::unordered_map<std::string, std::vector<Json::String>> dependencies_map;

    for (const auto& job : jobb.getMemberNames()) {
        std::vector<Json::String> dependencies;
        for (const auto& dependency : jobb[job]["dependencies"])
            dependencies.push_back(dependency.asString());

        dependencies_map[job] = dependencies;
    }

    std::unordered_set<std::string> visitedJobs;
    for (const std::string& job : jobs) 
        doingJob(job, dependencies_map, visitedJobs);

    std::cout << "Все работы выполнены!" << std::endl;
    return 0;
}