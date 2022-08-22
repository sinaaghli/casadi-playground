#include "foo/foo.h"
#include <casadi/casadi.hpp>
#include <iostream>
#include <string>

using namespace casadi;

void norm(){
  // Create scalar/matrix symbols
  MX x = MX::sym("x",5,1);

  // Compose into expressions
  MX y = norm_2(x);

  // Sensitivity of expression -> new expression
  MX grad_y = gradient(y,x);

  // Create a Function to evaluate expression
  Function f = Function("f",{x},{grad_y});

  // Evaluate numerically
  std::vector<DM> grad_y_num = f(DM({1,2,3,4,5}));

  std::cout << grad_y_num << std::endl;
}

void dynamics(){

  MX x = MX::sym("x",2); // Two states
  // Expression for ODE right-hand side
  MX z = 1-pow(x(1),2);
  MX rhs = vertcat(z*x(0)-x(1),x(0));

  MXDict ode;         // ODE declaration
  ode["x"]   = x;     // states
  ode["ode"] = rhs;   // right-hand side

  // Construct a Function that integrates over 4s
  Function F = integrator("F","cvodes",ode,{{"tf",4}});

  // Start from x=[0;1]
  DMDict res = F(DMDict{{"x0",std::vector<double>{0,1}}});

  // Sensitivity wrt initial state
  MXDict ress = F(MXDict{{"x0",x}});
  Function S("S",{x},{jacobian(ress["xf"],x)});
  std::cout << S(DM(std::vector<double>{0,1})) << std::endl;
}

void composition(){
  MX x = MX::sym("x",2); // Two states
  MX p = MX::sym("p");   // Free parameter

  // Expression for ODE right-hand side
  MX z = 1-pow(x(1),2);
  MX rhs = vertcat(z*x(0)-x(1)+2*tanh(p),x(0));
  MX tmp = 1+x(1)-2+2*x(1);
  std::cout << tmp << std::endl;
  tmp.simplify(tmp);
  std::cout << tmp.size() << std::endl;
  

  // ODE declaration with free parameter
  MXDict ode = {{"x",x},{"p",p},{"ode",rhs}};

  // Construct a Function that integrates over 1s
  Function F = integrator("F","cvodes",ode,{{"tf",1}});
  std::cout << doc_integrator("cvodes") << std::endl;

  // Control vector
  MX u = MX::sym("u",4,1);

  x = DM(std::vector<double>{0,1});  // Initial state
  for (int k=0;k<4;++k) {
    // Integrate 1s forward in time:
    // call integrator symbolically
    MXDict res = F({{"x0",x},{"p",u(k)}});
    x = res["xf"];
  }

  // NLP declaration
  MXDict nlp = {{"x",u},{"f",dot(u,u)},{"g",x}};

  // Solve using IPOPT
  // Function solver = nlpsol("solver","ipopt",nlp);
  // DMDict res = solver(DMDict{{"x0",0.2},{"lbg",0},{"ubg",0}});
}

class MyCallback : public Callback {
  double d;
public:
  MyCallback(const std::string& name, double d, const Dict& opts=Dict()) : d(d) {
    construct(name,opts);
  }
  ~MyCallback() override {}

  casadi_int get_n_in() override {return 2;}
  casadi_int get_n_out() override {return 1;}

  void init() override {
    std::cout << "initializing object" << std::endl;
  }

  std::vector<DM> eval(const std::vector<DM>& arg) const override {
    return {mtimes(arg.at(0),arg.at(1))};
  }
};

int main(int argc, char *argv[]) {
  // composition();

  // auto A = MX::sym("a",2,3);
  // auto B = MX::sym("b",3);
  // auto C = mtimes(A,B);
  // Function f = Function("f",SXIList(A,B),SXIList(mtimes(A,B)));

  SX x = SX::sym("x");
  SX y = SX::sym("y");
  SX z = SX::sym("z");
  y = x+1;
  z = 2*y;
  SX w = x+y+z;
  SX p = simplify(w);
  std::cout << w << std::endl;
  std::cout << p << std::endl;

  // auto I = DM(2,3);
  // auto J = DM(3);
  // MyCallback f("f",0.1);
  // MXVector args;
  // args.push_back(I);
  // args.push_back(J);
  // auto res = f(args);
  // std::cout << res.at(0) << std::endl;

  // std::string code
  // {"r[0] = x[0];\nwhile (r[0]<s[0]) {\nr[0] *= r[0];\n}"};
  // auto f = Function::jit("f",code,{"x","s"},{"r"});
  // std::cout << f << std::endl;
  
  // auto dae = DaeBuilder();
  // auto p1 = dae.add_p("p1");
  // auto u = dae.add_u("u");
  // auto x = dae.add_x("x");
  // dae.
  return 0;
}
