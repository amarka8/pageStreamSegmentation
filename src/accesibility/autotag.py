import os
import logging
import sys
import json
from pypdf import PdfReader
from pypdf import PdfWriter


from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices, ClientConfig
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.jobs.autotag_pdf_job import AutotagPDFJob
from adobe.pdfservices.operation.pdfjobs.params.autotag_pdf.autotag_pdf_params import AutotagPDFParams
from adobe.pdfservices.operation.pdfjobs.result.autotag_pdf_result import AutotagPDFResult


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def autotag_pdf_with_options(filename, client_id, client_secret):
    try:
        with open(filename, 'rb') as file:
            input_stream = file.read()
        

        # Initial setup, create credentials instance
        credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        client_config = ClientConfig(
            connect_timeout=8000,
            read_timeout=40000
        )

        # Creates a PDF Services instance
        pdf_services = PDFServices(credentials=credentials, client_config=client_config)

        # Creates an asset(s) from source file(s) and upload
        input_asset = pdf_services.upload(input_stream=input_stream,
                                        mime_type=PDFServicesMediaType.PDF)

        # Create parameters for the job
        autotag_pdf_params = AutotagPDFParams(
            generate_report=True,
            shift_headings=True
        )

        # Creates a new job instance
        autotag_pdf_job = AutotagPDFJob(input_asset=input_asset,
                                        autotag_pdf_params=autotag_pdf_params)

        # Submit the job and gets the job result
        location = pdf_services.submit(autotag_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, AutotagPDFResult)

        # Get content from the resulting asset(s)
        result_asset: CloudAsset = pdf_services_response.get_result().get_tagged_pdf()
        result_asset_report: CloudAsset = pdf_services_response.get_result().get_report()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        stream_asset_report: StreamAsset = pdf_services.get_content(result_asset_report)

        # Creates an output stream and copy stream asset's content to it
        os.makedirs("output/AutotagPDF", exist_ok=True)
        output_file_path = f"output/AutotagPDF/{filename}"
        output_file_path_report = f"output/AutotagPDF/{filename}.xlsx"

        with open(output_file_path, "wb") as file:
            file.write(stream_asset.get_input_stream())
        with open(output_file_path_report, "wb") as file:
            file.write(stream_asset_report.get_input_stream())

    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logging.exception(f'Filename : {filename} | Exception encountered while executing operation: {e}')

def add_viewer_preferences(pdf_path, filename):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    # Add all pages to the writer
    for page in reader.pages:
        writer.add_page(page)

    writer.create_viewer_preferences()
    writer.viewer_preferences.display_doctitle = True

    # Write the updated PDF to a file
    with open(filename, "wb") as f:
        writer.write(f)

    logger.info(f'Filename : {filename} | Viewer preferences added to the PDF')


def get_secret(basefilename):
    """
    Retrieves client credentials.
    
    Returns:
        tuple: (client_id, client_secret)
    """

    secret = '../../pdfservices-api-credentials.json'
    client_id = None
    client_secret = None


    try:
        with open(secret, 'r') as j:
            secret_dict = json.loads(j.read())        
        client_id = secret_dict['client_credentials']['client_id']
        client_secret = secret_dict['client_credentials']['client_secret']
    
    except Exception as e:
        logging.info(f'Filename : {basefilename} | Error with retrieving credentials: {e}')

    return client_id, client_secret


def process(file_path):
    """
    Main function that coordinates the downloading, processing, and uploading of PDF files and associated content.
    """

    try:    
        # bucket_name = os.getenv('S3_BUCKET_NAME')
        # file_key = os.getenv('S3_FILE_KEY').split('/')[2]
        # file_base_name = os.getenv('S3_FILE_KEY').split('/')[1]
        # logging.info(f'Filename : {file_key} | Bucket Name: {bucket_name}')
        # if not bucket_name or not file_key:
        #     logging.info("Error: S3_BUCKET_NAME and S3_FILE_KEY environment variables are required.")
        #     return

        # Define the local file path where the file will be saved
        # local_file_path = os.path.basename(file_path)  # Save the file with its original name
        
        # Download the file from S3
        # download_file_from_s3(bucket_name,file_base_name, file_key, local_file_path)

        # ex: ../../data/raw/KINGSTON-DOCUMENT-2021.pdf will get KINGSTON-DOCUMENT-2021.pdf
        base_filename = os.path.basename(file_path)
        # new filename which old file will be saved as
        filename = "COMPLIANT_" + base_filename

        client_id, client_secret = get_secret(base_filename)

        if not client_id or not client_secret:
            sys.exit(1)

        add_viewer_preferences(file_path, filename)

        autotag_pdf_with_options(filename, client_id, client_secret)

        # extract_api(filename, client_id, client_secret)

        # extract_api_zip_path = f"output/ExtractTextInfoFromPDF/extract${filename}.zip"
        # extract_to = f"output/zipfile/{filename}"
        # unzip_file(filename,extract_api_zip_path,extract_to)

    #     with open(f"output/zipfile/{filename}/structuredData.json") as file:
    #         data = json.load(file)

    #     pdf_document = pymupdf.open(filename)

    #     # Add TOC entries
    #     add_toc_to_pdf(filename,pdf_document,data)

    #     pdf_document.saveIncr()
    #     pdf_document.close()
    #     save_to_s3(filename, bucket_name, "output_autotag",file_base_name, file_key)

    #     logging.info(f"PDF saved with updated metadata and TOC. File location: COMPLIANT_{file_key}")

    #     figure_path = f"{extract_to}/figures"
    #     autotag_report_path = f"output/AutotagPDF/{filename}.xlsx"
    #     images_output_dir = "output/zipfile/images"

    #     s3_folder_autotag = f"temp/{file_base_name}/output_autotag"
    #     extract_images_from_excel(filename,figure_path,autotag_report_path,images_output_dir,bucket_name,s3_folder_autotag,file_key)
        
    #     logging.info(f'Filename : {file_key} | Processing completed for pdf file')
    except Exception as e:
        logger.info(f"File: {filename}, Status: Failed in First ECS task")
        logger.info(f"Filename : {file_path} | Error: {e}")
        sys.exit(1)

def main():
    process("../../data/raw/KINGSTON-DOCUMENT-2021.pdf")
        
if __name__ == "__main__":
    main()