use crate::data::Record;
use crate::features::{self, Preprocessor};
use crate::metrics::{self, ClassMetrics};
use ndarray::Array2;
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;

const PLOT_W: u32 = 1000;
const PLOT_H: u32 = 650;

/// Colour per grade (index aligns with `features::GRADES`).
const GRADE_COLORS: [RGBColor; 5] = [
    RGBColor(66, 133, 244),  // A - blue
    RGBColor(251, 188, 4),   // B - amber
    RGBColor(158, 158, 158), // C - grey
    RGBColor(239, 83, 80),   // D - red
    RGBColor(142, 36, 170),  // F - purple
];

const AXIS_FONT: &str = "sans-serif";
const TITLE_FONT: &str = "sans-serif";

fn drawing_area(path: &str, w: u32, h: u32) -> DrawingArea<SVGBackend<'_>, Shift> {
    SVGBackend::new(path, (w, h)).into_drawing_area()
}

fn into_plot_dir() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("outputs/plots")?;
    Ok(())
}

/// Build `n_bins` (bin_center, count) pairs within `lo..hi`.
fn histogram_bins(data: &[f64], n_bins: usize, lo: f64, hi: f64) -> (Vec<(f64, f64)>, f64) {
    let step = (hi - lo) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &v in data {
        let mut idx = ((v - lo) / step).floor() as isize;
        idx = idx.clamp(0, n_bins as isize - 1);
        counts[idx as usize] += 1;
    }
    (
        counts
            .into_iter()
            .enumerate()
            .map(|(i, c)| (lo + (i as f64 + 0.5) * step, c as f64))
            .collect(),
        step,
    )
}

fn min_max(data: &[f64]) -> (f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn draw_count_text(
    chart: &ChartContext<SVGBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    root: &DrawingArea<SVGBackend<'_>, Shift>,
    coord: (f64, f64),
    text: &str,
    style: TextStyle<'static>,
) -> Result<(), Box<dyn std::error::Error>> {
    let pos = chart.backend_coord(&coord);
    root.draw_text(text, &style, pos)?;
    Ok(())
}

/// Plot 1: bar chart of the number of students per final grade (shows class imbalance).
pub fn plot_grade_distribution(
    counts: &[usize; 5],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let max_count = *counts.iter().max().unwrap() as f64;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Final Grade Distribution - count of students per grade",
            (TITLE_FONT, 26).into_font(),
        )
        .margin(12)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..5f64, 0f64..(max_count * 1.18))?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(5)
        .x_label_formatter(&|&x| features::ordinal_to_grade(x as i32).to_string())
        .y_label_formatter(&|&y| format!("{}", y as usize))
        .x_desc("Final grade")
        .y_desc("Number of students")
        .axis_desc_style((AXIS_FONT, 18).into_font())
        .label_style((AXIS_FONT, 16).into_font())
        .draw()?;
    for (i, c) in counts.iter().enumerate() {
        chart
            .draw_series(std::iter::once(
                Rectangle::new(
                    [(i as f64 + 0.12, 0.0), (i as f64 + 0.88, *c as f64)],
                    GRADE_COLORS[i].mix(0.9).filled(),
                ),
            ))?;
        draw_count_text(
            &chart,
            &root,
            (i as f64 + 0.5, *c as f64 + 4.0),
            &c.to_string(),
            (AXIS_FONT, 18).into_font().color(&BLACK),
        )?;
    }
    root.present()?;
    Ok(())
}

/// Plot 2 & 5: 2x2 grid of histograms for the four numeric features, either
/// raw or z-scored depending on the title.
pub fn plot_numeric_histograms(
    columns: &[&[f64]],
    names: &[&str],
    title_prefix: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 2));
    let scaled = title_prefix.contains("Z-score");
    for (idx, area) in areas.iter().enumerate() {
        let data = columns[idx];
        let (lo, hi) = if scaled {
            (-4.0, 4.0)
        } else {
            let (lo, hi) = min_max(data);
            let pad = ((hi - lo) * 0.05).max(0.5);
            (lo - pad, hi + pad)
        };
        let (bins, step) = histogram_bins(data, 20, lo, hi);
        let max_y = bins.iter().map(|(_, y)| *y).fold(0.0f64, f64::max);
        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("{}: {}", title_prefix, names[idx]),
                (TITLE_FONT, 20).into_font(),
            )
            .margin(8)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(lo..hi, 0f64..(max_y * 1.1))?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(6)
            .y_labels(4)
            .y_label_formatter(&|&y| format!("{:.0}", y))
            .x_desc(if scaled { "z-score" } else { "value" })
            .y_desc("count")
            .axis_desc_style((AXIS_FONT, 13).into_font())
            .label_style((AXIS_FONT, 12).into_font())
            .draw()?;
        for (center, count) in bins {
            chart
                .draw_series(std::iter::once(
                    Rectangle::new(
                        [
                            (center - step * 0.45, 0.0),
                            (center + step * 0.45, count),
                        ],
                        RGBColor(52, 168, 83).mix(0.85).filled(),
                    )
                    ))?;
        }
    }
    root.present()?;
    Ok(())
}

/// Plot 3: attendance vs study time scatter, one point series per grade.
pub fn plot_scatter_by_grade(
    points: &[Vec<(f64, f64)>],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let mut all_x = Vec::new();
    let mut all_y = Vec::new();
    for pts in points {
        for (x, y) in pts {
            all_x.push(*x);
            all_y.push(*y);
        }
    }
    let (x_lo, x_hi) = min_max(&all_x);
    let (y_lo, y_hi) = min_max(&all_y);
    let pad = 1.0;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Attendance % vs Study Time (hours) - points coloured by final grade",
            (TITLE_FONT, 24).into_font(),
        )
        .margin(12)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d((x_lo - pad)..(x_hi + pad), (y_lo - pad)..(y_hi + pad))?;
    chart
        .configure_mesh()
        .x_desc("Attendance percent")
        .y_desc("Study time (hours)")
        .axis_desc_style((AXIS_FONT, 18).into_font())
        .label_style((AXIS_FONT, 15).into_font())
        .draw()?;
    for (i, pts) in points.iter().enumerate() {
        let color = GRADE_COLORS[i];
        chart
            .draw_series(PointSeries::of_element(
                pts.clone(),
                4,
                &color,
                &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
            ))?
            .label(features::GRADES[i].to_string())
            .legend(move |(x, y)| Circle::new((x + 5, y), 5, color.filled()));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font((AXIS_FONT, 15))
        .draw()?;
    root.present()?;
    Ok(())
}

/// Plot 4: mean +/- std of previous_grade for each final grade (separation power).
pub fn plot_previous_grade_by_grade(
    means: &[f64],
    stds: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let max_val = means
        .iter()
        .zip(stds)
        .map(|(m, s)| m + s)
        .fold(0.0f64, f64::max);
    let min_val = means
        .iter()
        .zip(stds)
        .map(|(m, s)| m - s)
        .fold(f64::INFINITY, f64::min);
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Mean previous grade (+/- 1 std) by final grade - how much the feature separates classes",
            (TITLE_FONT, 22).into_font(),
        )
        .margin(12)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..5f64, (min_val - 2.0)..(max_val + 2.0))?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(5)
        .x_label_formatter(&|&x| features::ordinal_to_grade(x as i32).to_string())
        .x_desc("Final grade")
        .y_desc("Previous grade (mean +/- std)")
        .axis_desc_style((AXIS_FONT, 18).into_font())
        .label_style((AXIS_FONT, 15).into_font())
        .draw()?;
    for (i, (m, s)) in means.iter().zip(stds).enumerate() {
        chart
            .draw_series(std::iter::once(
                Rectangle::new(
                    [(i as f64 + 0.2, 0.0), (i as f64 + 0.8, *m)],
                    GRADE_COLORS[i].mix(0.7).filled(),
                )
                ))?;
        chart
            .draw_series(std::iter::once(ErrorBar::new_vertical(
                i as f64 + 0.5,
                m - s,
                *m,
                m + s,
                BLACK.stroke_width(3),
                12,
            )))?;
    }
    root.present()?;
    Ok(())
}

/// Plot 6: confusion matrix as a coloured heatmap (rows = actual, cols = predicted).
pub fn plot_confusion_matrix(
    cm: &Array2<usize>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = cm.nrows();
    let max_v = *cm.iter().max().unwrap() as f64;
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Confusion Matrix on the test set (rows = actual grade, columns = predicted grade)",
            (TITLE_FONT, 24).into_font(),
        )
        .margin(12)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..n as f64, 0f64..n as f64)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(n)
        .y_labels(n)
        .x_label_formatter(&|&x| features::ordinal_to_grade(x as i32).to_string())
        .y_label_formatter(&|&y| features::ordinal_to_grade(y as i32).to_string())
        .label_style((AXIS_FONT, 18).into_font())
        .draw()?;
    for i in 0..n {
        for j in 0..n {
            let v = cm[[i, j]];
            let intensity = if max_v == 0.0 { 0.0 } else { v as f64 / max_v };
            let color = HSLColor(240.0, 0.75, 0.96 - 0.55 * intensity);
            chart
                .draw_series(std::iter::once(
                    Rectangle::new(
                        [
                            (j as f64 + 0.03, i as f64 + 0.03),
                            (j as f64 + 0.97, i as f64 + 0.97),
                        ],
                        color.filled(),
                    ),
                ))?;
            let text_color = if intensity > 0.6 { WHITE } else { BLACK };
            draw_count_text(
                &chart,
                &root,
                (j as f64 + 0.5, i as f64 + 0.5),
                &v.to_string(),
                (AXIS_FONT, 20).into_font().color(&text_color),
            )?;
        }
    }
    root.present()?;
    Ok(())
}

fn plot_metric_bars(
    values: &[f64],
    title: &str,
    color: RGBColor,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, (TITLE_FONT, 22).into_font())
        .margin(12)
        .x_label_area_size(55)
        .y_label_area_size(55)
        .build_cartesian_2d(0f64..5f64, 0f64..1.05f64)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(5)
        .x_label_formatter(&|&x| {
            let idx = x as i32;
            if (0..5).contains(&idx) {
                features::ordinal_to_grade(idx).to_string()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|&y| format!("{:.2}", y))
        .x_desc("Grade")
        .axis_desc_style((AXIS_FONT, 16).into_font())
        .label_style((AXIS_FONT, 15).into_font())
        .draw()?;
    for (i, v) in values.iter().enumerate() {
        chart
            .draw_series(std::iter::once(
                Rectangle::new(
                    [(i as f64 + 0.2, 0.0), (i as f64 + 0.8, *v)],
                    color.mix(0.85).filled(),
                )
                ))?;
        draw_count_text(
            &chart,
            &root,
            (i as f64 + 0.5, v + 0.02),
            &format!("{:.3}", v),
            (AXIS_FONT, 14).into_font().color(&BLACK),
        )?;
    }
    root.present()?;
    Ok(())
}

/// Plot 7: precision, recall and F1 per class as three bar charts.
pub fn plot_per_class_metrics(
    metrics: &[(i32, ClassMetrics)],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let precision: Vec<f64> = metrics.iter().map(|(_, m)| m.precision).collect();
    let recall: Vec<f64> = metrics.iter().map(|(_, m)| m.recall).collect();
    let f1: Vec<f64> = metrics.iter().map(|(_, m)| m.f1).collect();
    let base = path.trim_end_matches(".svg");
    plot_metric_bars(
        &precision,
        "Precision per grade (of the predicted positives, how many are correct)",
        RGBColor(66, 133, 244),
        &format!("{base}.svg"),
    )?;
    plot_metric_bars(
        &recall,
        "Recall per grade (of the actual class, how many were found)",
        RGBColor(251, 188, 4),
        &format!("{base}_recall.svg"),
    )?;
    plot_metric_bars(
        &f1,
        "F1 score per grade (harmonic mean of precision and recall)",
        RGBColor(52, 168, 83),
        &format!("{base}_f1.svg"),
    )?;
    Ok(())
}

/// Plot 8: mean absolute model weight per feature (proxy for feature importance).
pub fn plot_feature_importance(
    names: &[String],
    weights: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = names.len();
    let mut pairs: Vec<(&str, f64)> = names.iter().map(|s| s.as_str()).zip(weights).map(|(n, w)| (n, *w)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let max_w = weights.iter().cloned().fold(0.0f64, f64::max).max(1e-6);
    let root = drawing_area(path, PLOT_W, PLOT_H);
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Mean absolute logistic-regression weight per feature - what the model relies on",
            (TITLE_FONT, 22).into_font(),
        )
        .margin(12)
        .x_label_area_size(70)
        .y_label_area_size(240)
        .build_cartesian_2d(0f64..(max_w * 1.08), 0f64..n as f64)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(6)
        .y_labels(n)
        .y_label_formatter(&|&y| {
            let idx = y.round() as usize;
            if idx < n {
                pairs[idx].0.to_string()
            } else {
                String::new()
            }
        })
        .x_desc("Mean |weight| across the 5 classes")
        .axis_desc_style((AXIS_FONT, 15).into_font())
        .label_style((AXIS_FONT, 14).into_font())
        .draw()?;
    for (i, (_, w)) in pairs.iter().enumerate() {
        chart
            .draw_series(std::iter::once(
                Rectangle::new(
                    [(0.0, i as f64 + 0.15), (*w, i as f64 + 0.85)],
                    RGBColor(214, 92, 40).mix(0.85).filled(),
                )
                ))?;
    }
    root.present()?;
    Ok(())
}

/// Render every labelled plot under `outputs/plots/`.
pub fn render_all(
    records: &[Record],
    pre: &Preprocessor,
    cm: &Array2<usize>,
    model: &linfa_logistic::MultiFittedLogisticRegression<f64, i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    into_plot_dir()?;

    // 1. Grade distribution
    let mut counts = [0usize; 5];
    for r in records {
        counts[features::grade_to_ordinal(&r.final_grade) as usize] += 1;
    }
    plot_grade_distribution(&counts, "outputs/plots/01_grade_distribution.svg")?;

    // 2. Raw numeric feature histograms
    let numeric: Vec<Vec<f64>> = (0..4)
        .map(|i| numeric_column(records, i))
        .collect();
    let numeric_refs: Vec<&[f64]> = numeric.iter().map(|v| v.as_slice()).collect();
    plot_numeric_histograms(
        &numeric_refs,
        &features::NUMERIC_FEATURES,
        "Raw distribution",
        "outputs/plots/02_numeric_feature_histograms.svg",
    )?;

    // 3. Scatter by grade
    let mut points: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 5];
    for r in records {
        let ord = features::grade_to_ordinal(&r.final_grade) as usize;
        points[ord].push((r.attendance_percent, r.study_time_hours));
    }
    plot_scatter_by_grade(
        &points,
        "outputs/plots/03_attendance_vs_study_time_scatter.svg",
    )?;

    // 4. Mean previous grade by final grade
    let mut sums = [0.0f64; 5];
    let mut sumsq = [0.0f64; 5];
    let mut ns = [0usize; 5];
    for r in records {
        let ord = features::grade_to_ordinal(&r.final_grade) as usize;
        sums[ord] += r.previous_grade;
        sumsq[ord] += r.previous_grade * r.previous_grade;
        ns[ord] += 1;
    }
    let means: Vec<f64> = (0..5).map(|i| sums[i] / ns[i] as f64).collect();
    let stds: Vec<f64> = (0..5)
        .map(|i| {
            let var = (sumsq[i] / ns[i] as f64) - means[i] * means[i];
            var.max(0.0).sqrt()
        })
        .collect();
    plot_previous_grade_by_grade(
        &means,
        &stds,
        "outputs/plots/04_previous_grade_by_final_grade.svg",
    )?;

    // 5. Z-scored numeric histograms
    let scaled: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            records
                .iter()
                .map(|r| (numeric_value(r, i) - pre.numeric_mean[i]) / pre.numeric_std[i])
                .collect()
        })
        .collect();
    let scaled_refs: Vec<&[f64]> = scaled.iter().map(|v| v.as_slice()).collect();
    plot_numeric_histograms(
        &scaled_refs,
        &features::NUMERIC_FEATURES,
        "After Z-score normalization",
        "outputs/plots/05_scaled_feature_histograms.svg",
    )?;

    // 6. Confusion matrix
    plot_confusion_matrix(cm, "outputs/plots/06_confusion_matrix.svg")?;

    // 7. Per-class metrics
    let per_class = metrics::per_class_metrics(cm, &[0, 1, 2, 3, 4]);
    plot_per_class_metrics(&per_class, "outputs/plots/07_per_class_metrics.svg")?;

    // 8. Feature importance from model weights
    let params = model.params();
    // params shape: (n_features, n_classes) -> mean abs across classes
    let importances: Vec<f64> = params
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>() / params.ncols() as f64)
        .collect();
    plot_feature_importance(
        &pre.feature_names,
        &importances,
        "outputs/plots/08_feature_importance.svg",
    )?;

    Ok(())
}

fn numeric_column(records: &[Record], i: usize) -> Vec<f64> {
    records.iter().map(|r| numeric_value(r, i)).collect()
}

fn numeric_value(r: &Record, i: usize) -> f64 {
    match i {
        0 => r.study_time_hours,
        1 => r.attendance_percent,
        2 => r.sleep_hours,
        _ => r.previous_grade,
    }
}
