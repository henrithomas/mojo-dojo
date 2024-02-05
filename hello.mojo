
def greet(name):
    return "hello, " + name

fn greet_strong(name: String, x: Int = 4) -> String:
    return "hello, " + name + " " + (2 * x)

fn add(x: Int = 1, y: Int = 1) -> Int:
    return x + y

fn main():
    let name = "henri"
    print("hello, world!")
    let test = greet_strong(name)
    print(test)
    let test2 = greet_strong(name, 5)
    print(test2)
    let test3 = add(4,4)
    print(test3)
    print(add(x=6, y=45))