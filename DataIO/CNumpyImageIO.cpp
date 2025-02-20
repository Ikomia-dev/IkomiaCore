// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CNumpyImageIO.h"
#include "Data/CDataImageInfo.h"
#include "Data/CvMatNumpyArrayConverter.h"

using namespace boost::python;

CNumpyImageIO::CNumpyImageIO(const std::string &fileName) : CVirtualImageIO(fileName)
{
    loadNumpyArray();
}

CNumpyImageIO::~CNumpyImageIO()
{
}

VectorString CNumpyImageIO::fileNames(const SubsetBounds &bounds)
{
    Q_UNUSED(bounds);
    VectorString files = {m_fileName};
    return files;
}

Dimensions CNumpyImageIO::dimensions()
{
    return dimensions(SubsetBounds());
}

Dimensions CNumpyImageIO::dimensions(const SubsetBounds &bounds)
{
    Q_UNUSED(bounds);

    // Manage only [N H W] or [H W] arrays
    if(m_dims.size() == 2)
        return {{DataDimension::IMAGE, 1}};
    else if(m_dims.size() == 3)
        return {{DataDimension::IMAGE, m_dims[0]}};
    else
        return {};
}

CDataInfoPtr CNumpyImageIO::dataInfo()
{
    return dataInfo(SubsetBounds());
}

CDataInfoPtr CNumpyImageIO::dataInfo(const SubsetBounds &bounds)
{
    Q_UNUSED(bounds);

    // Manage only [N H W] or [H W] arrays
    auto infoPtr = std::make_shared<CDataImageInfo>();
    infoPtr->setFileName(m_fileName);

    if(m_dims.size() == 2)
    {
        infoPtr->m_height = m_dims[0];
        infoPtr->m_width = m_dims[1];
    }
    else if(m_dims.size() == 3)
    {
        infoPtr->m_height = m_dims[1];
        infoPtr->m_width = m_dims[2];
    }
    return infoPtr;
}

CMat CNumpyImageIO::read()
{
    return read(SubsetBounds());
}

CMat CNumpyImageIO::read(const SubsetBounds &bounds)
{
    Q_UNUSED(bounds);

    if(m_images.size() == 1)
        return m_images[0];
    else if(m_images.size() > 0)
    {
        // TODO manage bounds correctly
        // For now, we create a merge image (to fit the use case of mask segmentation)
        int h = m_dims[1];
        int w = m_dims[2];
        cv::Mat labelImg = cv::Mat::zeros(h, w, CV_8UC1);

        for(size_t i=0; i<m_images.size(); ++i)
        {
            cv::Mat mask;
            m_images[i].convertTo(mask, CV_8U);
            CMat maskValue(h, w, CV_8UC1, cv::Scalar(i + 1));
            CMat maskLabel(h, w, CV_8UC1, cv::Scalar(0));
            maskValue.copyTo(maskLabel, mask);
            //Merge label mask to final label image
            cv::bitwise_or(labelImg, maskLabel, labelImg);
        }
        return labelImg;
    }
    else
        return CMat();
}

void CNumpyImageIO::loadNumpyArray()
{
    try
    {
        CPyEnsureGIL gil;
        BoostNumpyArrayToCvMatConverter();

        str path(m_fileName);
        object dataModule = import("ikomia.utils.data");
        object numpyImg = dataModule.attr("NumpyImage")(path, "arr_0");

        // Set dim order, should be manage dynamically
        tuple dimOrder = make_tuple(2, 0, 1);
        numpyImg.attr("set_dim_order")(dimOrder);

        // Get array dims
        tuple arrayDims = extract<tuple>(numpyImg.attr("get_dims")());
        for(int i=0; i<len(arrayDims); ++i)
            m_dims.push_back(extract<int>(arrayDims[i]));

        // Get data
        if(m_dims.size() == 2)
            m_images.push_back(extract<CMat>(numpyImg.attr("get_data")()));
        else if(m_dims.size() == 3)
        {
            for(int i=0; i<m_dims[0]; ++i)
                m_images.push_back(extract<CMat>(numpyImg.attr("get_2d_data")(i)));
        }
    }
    catch(error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException());
    }
}
