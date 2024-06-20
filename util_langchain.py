# 安装 langchain 后要额外安装 pip install pymupdf rapidocr-onnxruntime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
# 要额外安装 pip install unstructured
# Resource punkt not found.
# Please use the NLTK Downloader to obtain the resource: nltk.download('punkt')，需要魔术方法，下载到 C:\Users\admin\AppData\Roaming\nltk_data 中
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document

from util import *


# 获取 pdfs 逐页数据，可以选择是否对图片解析，注意是解析图片中的文字，而不返回图片：extract_from_images_with_rapidocr
# 实际测试下来，增加图片解析会极大的减慢速度，而且效果也不是很好
def get_pdf_pages(_file_path, _extract_images: bool = False, _clean=True) -> List[Document]:
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdfs 文档路径
    loader = PyMuPDFLoader(file_path=_file_path, extract_images=_extract_images)

    # 调用 PyMuPDFLoader Class 的函数 load 对 pdfs 文件进行加载
    # 返回 list, 每一个元素是 langchain_core.documents.base.Document
    page_list = loader.load()
    if _clean:
        # 1. 统一换行符为 \n
        # 2. 删除额外的回车，langchain 会在原本两个符号中间插入了 \n
        # 。\n“...”\n。 -> 。“...”。
        # 3. 删除所有汉字之间的空格
        # 4. 将剩余的多个连续空格都变成一个空格
        for page in page_list:
            page.page_content = replace_multiple_spaces(
                delete_all_spaces_between_chinese(delete_one_extra_break_line(replace_break_line(page.page_content))))

        # TODO：目前还无法处理下面这种换行和段落的区别
        # 前言
        # “周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读
        # 者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推
        # 导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充
        # 具体的推导细节。”
        # 读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周

    return page_list


# 获取 markdown 逐页数据，single 表示整体会返回一个 Document
def get_md_pages(_file_path, _mode: str = "single", _clean=True) -> List[Document]:
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdfs 文档路径
    loader = UnstructuredMarkdownLoader(file_path=_file_path, mode=_mode)

    # 返回 list, 每一个元素是 langchain_core.documents.base.Document
    md_list = loader.load()

    if _clean:
        # 1. md文件每一段中间隔了一个换行符，我们同样可以使用replace方法去除。
        for md in md_list:
            md.page_content = md.page_content.replace('\n\n', '\n')

    return md_list


def check_get_pdf_pages():
    pdf_pages = get_pdf_pages(BIGDATA_PDF_PATH + "pumpkin_book.pdf")
    assert_equal(len(pdf_pages), 196)

    pdf_page = pdf_pages[16]
    # 公式无法很好的解析
    print(f"该文档的描述性数据：{pdf_page.metadata}",
          f"查看该文档的内容:\n{pdf_page.page_content}",
          sep="\n------\n")

    pdf_page = pdf_pages[1]
    # print 可以增加 sep 分隔符
    print(f"该文档的描述性数据：{pdf_page.metadata}",
          f"查看该文档的内容:\n{pdf_page.page_content}",
          sep="\n------\n")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50
    )
    split_text_list = text_splitter.split_text(pdf_page.page_content)
    # 分割前文档一共有 1312 个文字.
    # 分割后，一共有 3 段，每段文字分别为 [497, 498, 314]，总文字数是 1309.
    print(F"分割前文档一共有 {len(pdf_page.page_content)} 个文字.")
    print(
        F"分割后，一共有 {len(split_text_list)} 段，每段文字分别为 {[len(text) for text in split_text_list]}，总文字数是 {sum(len(text) for text in split_text_list)}.")

    print_list(split_text_list)


def check_get_md_pages():
    md_page = get_md_pages(BIGDATA_MD_PATH + "1. 简介 Introduction.md")
    assert_equal(len(md_page), 1)

    md_page = md_page[0]
    # 公式无法很好的解析
    print(f"该文档的描述性数据：{md_page.metadata}",
          f"查看该文档的内容:\n{md_page.page_content}",
          sep="\n------\n")


@func_timer(arg=True)
def main():
    assert_equal(len("\n"), 1)
    assert_equal(len("\n\n"), 2)
    assert_equal(len("  \n \n"), 5)

    check_get_pdf_pages()
    check_get_md_pages()


if __name__ == '__main__':
    main()
