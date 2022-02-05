import os
import pefile

def extract_infos(fpath):
    res = []
    res.append(os.path.basename(fpath))
    pe = pefile.PE(fpath)
    res.append(pe.FILE_HEADER.Machine)
    res.append(pe.FILE_HEADER.SizeOfOptionalHeader)
    res.append(pe.FILE_HEADER.Characteristics)


if __name__ == '__main__':

    # Launch legitimate
    for ffile in os.listdir('legitimate'):
        print(ffile)
        try:
            res = extract_infos(os.path.join('legitimate/', ffile))
        except pefile.PEFormatError:
            print('\t -> Bad PE format - Deleting')
            try:
                os.remove(ffile)
            except FileNotFoundError:
                print('a')

    for ffile in os.listdir('malicious'):
        print(ffile)
        try:
            res = extract_infos(os.path.join('malicious/', ffile))
        except pefile.PEFormatError:
            print('\t -> Bad PE format - Deleting')
            try:
                os.remove(ffile)
            except FileNotFoundError:
                print('a')

        except:
            print('\t -> Weird error')
