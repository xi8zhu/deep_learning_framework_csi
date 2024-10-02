
def data_split_validate(string):
    try:
        parts = string.split(':')
        if len(parts) != 3:
            raise ValueError("format error: data split string!")
        numbers = list(map(int, parts))

        if sum(numbers) != 10:
            raise ValueError("format error: the sum of three number in data split string is not 10!")
        numbers = [i/10 for i in numbers]
        return numbers
    
    except ValueError as e:
        raise ValueError(f"input error: {e}")

if __name__ == '__main__':
    result = data_split_validate('9:1:0')
    print(result)
    
