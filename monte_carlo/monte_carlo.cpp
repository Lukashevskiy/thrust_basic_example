#include <algorithm>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <format>
#include <limits>
#include <string>
#include <vector>
#include <random>
#include <chrono>

struct trapesoid_t{
    double left, right, upper, lower;
    trapesoid_t(double l, double r, double u, double d): left(l), right(r), lower(u), upper(d){}
};

int random_value(double l, double r){
    std::random_device rd;  
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(l, r);

    return distrib(gen);
}

double monte_carlo_area(size_t num_samples, std::function<double(double)> upper_line_func, double left, double right, double upper=std::numeric_limits<double>::min(), double down=0){
    trapesoid_t t(left, right, upper, down);

    for(int i = 0; i != num_samples; ++i){
        int value = random_value(t.left, t.right);
        t.upper = std::max(t.upper, upper_line_func(value));
    }

    size_t inner_points{};
    for(int i = 0; i != num_samples; ++i){
        int x = random_value(t.left, t.right);
        int y = random_value(t.left, t.right);

        if (y <= upper_line_func(x)){
            ++inner_points;
        }
    }


    size_t rect_area = (t.right - t.left) * t.upper;
    size_t area = rect_area * inner_points / num_samples;
    return area;
}

int main(int argc, char *argv[]){
    double a = std::stod(argv[1]);
    double b = std::stod(argv[2]);

    auto upper_line_func = [](double x){return (x * x)/(x + 1) + 1 / x;};

    std::vector<size_t> arr_num_samples{100, 10'000, 100'000, 1'000'000};
    for(auto &&num_samples: arr_num_samples){
        
        auto time_begin = std::chrono::high_resolution_clock().now();
        double area = monte_carlo_area(num_samples, upper_line_func, a, b);
        auto time_end = std::chrono::high_resolution_clock().now();

        auto duration_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin);

        std::cout << "---------------------------------------\n";
        std::cout << std::format("Num samples \t- {}\nTrapesoid \t- {}\n", num_samples, area);
        std::cout << std::format("TIME DURATION \t- {} milliseconds\n", duration_time.count());
        std::cout << "---------------------------------------\n";
    }
}