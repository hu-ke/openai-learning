import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// const loader = new PDFLoader("../thethreekingdoms.pdf");
// const loader = new PDFLoader('../nodejs.pdf')
const loader = new PDFLoader('../zhixingheyi.pdf')
// const loader = new PDFLoader('../charlie-and-the-chocolate-factory-by-roald-dahl.pdf')

const docs = await loader.load();
console.log(docs)