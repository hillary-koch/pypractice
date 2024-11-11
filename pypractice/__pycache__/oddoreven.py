
def even_or_odd(input_num):
    try:
        modulo = int(input_num) % 2
    except:
        print('You didn\'t pass an integer. Please try again.')
        prompt = 'What number do you want to know about? '
        input_num = input(prompt)
        even_or_odd(input_num)
    else:
        if modulo == 0:
            if int(input_num) % 4 == 0:
                print(f'{input_num} is even and ALSO divisible by 4.')    
            else:
                print(f'{input_num} is even.')
                
        elif(modulo == 1):
            print(f'{input_num} is odd.')

if __name__ == '__main__':    
    prompt = 'What number do you want to know about? '
    input_num = input(prompt)
    even_or_odd(input_num)
