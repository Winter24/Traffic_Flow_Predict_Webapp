#Before running this file make sure you remove all the html <div> just in case 
import re

with open('/home/winter24/Documents/python code/DAP/Traffic_Flow_Predict_Webapp/div_contents.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Regular expressions to find content after "name" and "id\\u003d"
name_content = re.findall(r'\[\\"name\\",\[\\"(.*?)\\"', content)
id_content = re.findall(r'id\\\\u003d([a-f0-9]+)', content)

# Remove the first three elements from id_content
id_content = id_content[3:]

# Combine the extracted 'name' and 'id' content into a single list of tuples
combined_content = list(zip(name_content, id_content))


output_file_path = './id.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for name, id_val in combined_content:
        file.write(f"{name}: {id_val}\n")
